# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn

import glob
import logging
import os
import time
import fcntl
import math
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import PreTrainedModel, T5Model
#from transformers.utils import ModelOutput
#from transformers.tokenization_utils import trim_batch

from base_transformer import BaseTransformer, add_generic_args, generic_train
from data import KiltDataset, seq2seq_to_kilt, dataset_config, remove_preds
from kilt.eval_downstream import normalize_answer
# from kilt.kilt_utils import utils
import kilt.kilt_utils as utils
from typing import Callable, List, Optional, Tuple, Union

from kilt.retrievers import DPR_connector
from kilt.retrievers.dpr.dense_retriever import generate_sparse_inputs
from kilt.retrievers.DPR_connector import get_representation_tensor, sparse_vector_to_dict, dict_sparse_to_string, load_list_from_file, calculate_flop_reg

from transformers import LogitsProcessorList, BeamSearchScorer, PretrainedConfig,StoppingCriteriaList
# from ...modeling_outputs import ModelOutput
# from ...modeling_utils import PreTrainedModel
from kilt.knowledge_source import KnowledgeSource
from splade.src.models.transformer_rep import Splade
from pytorch_lightning.plugins import DDPPlugin

import random

logger = logging.getLogger(__name__)

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])




class Seq2seqTransformer(BaseTransformer):
    def __init__(self, hparams):
        super().__init__(hparams, num_labels=None)
        self.lr_scheduler = None
        self.devsets = {}
        self.em = -1
        self.dataset_list = self.hparams.dataset.split(',')
        self.eval_batch_size = 100000
        self.train_batch_size = 100000
        self.source_length = -1
        self.target_length = -1
        # Maintance epoch count additionally
        self.my_epoch_count = 0
 

        special_tokens = []

        for i in range(0, 101):
            special_tokens.append('<extra_id_' + str(i) + '>')

        special_tokens.extend(['[START_ENT]', '[END_ENT]', 'Question Answering:', 'Entity Linking:',
                               'Fact Checking:', 'Dialogue:', 'Relation Extraction:', '[SEP]'])
        self.tokenizer.sep_token_id = self.tokenizer.sep_token_id
        
        # self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)


        fevers_classes = ["<SUPPORTS>", "<REFUTES>"]

        self.tokenizer.add_tokens(fevers_classes, special_tokens=True)

        self.model.resize_token_embeddings(len(self.tokenizer))


        last_length = len(self.tokenizer)

        #Add Soft-Prompt
        if hparams.use_prompt:
            for param in self.model.parameters():
                param.requires_grad = False
            Task_type = ['Question Answering', 'Entity Linking', 'Fact Checking', 'Dialogue', 'Relation Extraction','Slot Filling']
            prompt_tokens = []
            for t in Task_type:
                for i in range(100):
                    prompt_tokens.append(t+'#'+str(i))
            
            self.tokenizer.add_tokens(prompt_tokens, special_tokens=True)

            self.model.resize_token_embeddings(len(self.tokenizer))

            self.model.set_input_embeddings(DPR_connector.SoftEmbedding(self.model.get_input_embeddings(), hparams.soft_prompt_length))


        self.model.config.n_positions = hparams.max_sequence_length
        
        # Model is eliminated
        self.model = None

        self.bad_words = [[self.tokenizer.convert_tokens_to_ids(bad_word)] for bad_word in
                          self.tokenizer.additional_special_tokens]
        self.eos_token = self.tokenizer.eos_token
        
        #print(dataset_config.keys())
        #print(dict(dataset_config))
        for d in self.dataset_list:
            train_batch = int(hparams['train_batch'])
            eval_batch = int(hparams['eval_batch'])
            source_length = int(hparams.datasets[d]['source_length']) if hparams.datasets.get(d) is not None else 512
            target_length = int(hparams.datasets[d]['target_length']) if hparams.datasets.get(d) is not None else 512

            if train_batch < self.train_batch_size:
                self.train_batch_size = train_batch
            if eval_batch < self.eval_batch_size:
                self.eval_batch_size = eval_batch
            if source_length > self.source_length:
                self.source_length = source_length
            if target_length > self.target_length:
                self.target_length = target_length

        self.data_dir = self.hparams.data_dir
        self.output_dir = self.hparams.output_dir

        # Initilize retriever
        self.retriever = DPR_connector.DPR(hparams.model_name, hparams)
        self.after_forked = True # Use to avoid fork some thing
        if not hparams.debug_mode:
            if self.hparams.dense_model:
                self.question_encoder = self.retriever.encoder # register question decoder there
                self.ctx_encoder = self.retriever.ctx_encoder

                self.dpr_tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_type)
            else:
                self.hparams.last_vec_path += '_sparse' # avoid override dense save
                sparse_model_name = 'naver/splade-cocondenser-ensembledistil'
                self.question_encoder = Splade(sparse_model_name, agg="max")
                self.dpr_tokenizer = AutoTokenizer.from_pretrained(sparse_model_name)
                self.vocab_id2token = dict((i, s) for s, i in self.dpr_tokenizer.get_vocab().items())
                self.ctx_encoder = Splade(sparse_model_name, agg="max")

            # Freeze weight of ctx_encoder
            for p in self.ctx_encoder.parameters():
                p.requires_grad = False
                
                # self.question_encoder = self.
        self.vec_dropout = nn.Dropout(0.2)

        self.ks = KnowledgeSource()

        self.save_hyperparameters()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def generate_vec(self, text_list):
        # inputs = self.dpr_tokenizer.batch_encode_plus(
        #     text_list, add_special_tokens=True, max_length=self.hparams.max_sequence_length, pad_to_max_length=True,
        #     return_tensors="pt",
        #     return_attention_mask=True,
        #     truncation=True,
        # )
        # for k in inputs.keys():
        #     inputs[k] = inputs[k].cuda()
        inputs = generate_sparse_inputs(self.question_encoder, self.dpr_tokenizer, text_list)
        return self.question_encoder(q_kwargs=inputs)["q_rep"]
        
    
    def gen_vec(self, ns, gold, query):
        ns_length = len(ns)
        text_list = ns + gold

        if self.hparams.dense_model:
            ret = self.retriever.retriever.generate_context_vectors(text_list, self.ctx_encoder)
            query_output = self.retriever.retriever.generate_question_vectors(query)
        else:
            context_inputs = generate_sparse_inputs(self.ctx_encoder, self.dpr_tokenizer, text_list)
            query_inputs = generate_sparse_inputs(self.question_encoder, self.dpr_tokenizer, query)
            # context_inputs = self.dpr_tokenizer.batch_encode_plus(
            #     text_list, add_special_tokens=True, max_length=self.hparams.max_sequence_length, pad_to_max_length=True,
            #     return_tensors="pt",
            #     return_attention_mask=True,
            #     truncation=True,
            # )
            # query_inputs = self.dpr_tokenizer.batch_encode_plus(
            #     query, add_special_tokens=True, max_length=self.hparams.max_sequence_length, pad_to_max_length=True,
            #     return_tensors="pt",
            #     return_attention_mask=True,
            #     truncation=True,
            # )


            # for k in context_inputs.keys():
            #     context_inputs[k] = context_inputs[k].cuda()
            # for k in query_inputs.keys():
            #     query_inputs[k] = query_inputs[k].cuda()
            with torch.no_grad():
                ret = self.ctx_encoder(d_kwargs=context_inputs)["d_rep"]
            query_output = self.question_encoder(q_kwargs=query_inputs)["q_rep"]
        
        return ret[:ns_length], ret[ns_length:], query_output

    def _step(self, batch):
        gp = []
        for p, qid in zip(batch["gold_provenance"],batch['query_ids']):
            if p.get('start_paragraph_id') is not None:
                # Use for dataset including fever
                page = self.ks.get_page_by_id(p['wikipedia_id'])
                pas = page['text'][p['start_paragraph_id']]
            elif self.retriever.title_to_text.get(p['title']):
                pas = self.retriever.title_to_text[p['title']]
            else:
                page = self.ks.get_page_by_id(p['wikipedia_id'])
                pas = page['text'][0]
            gp.append(pas)
        # print('pas:',pas)
        ns_p = []
        for p in batch["negative_samples"]:
            for tp in p:
                ns_p.append(tp["text"])
                # print('tp_text:', tp["text"])

        ns_vec, gold_vec, query_vec = self.gen_vec(ns_p, gp, batch['question'])
        ns_vec = ns_vec.view(gold_vec.size(0), -1, gold_vec.size(-1))
        gold_vec = gold_vec.unsqueeze(1)
        query_vec = query_vec.unsqueeze(1)
        sz = math.sqrt(gold_vec.size(-1))
        scores = -torch.log_softmax(torch.sum(torch.cat([ns_vec, gold_vec], 1) * query_vec,-1) / sz, -1)
        loss = scores[:, -1]
        if not self.hparams.dense_model:
            loss = loss + 1e-4 * calculate_flop_reg(query_vec)
        return loss, query_vec.detach().cpu(), batch["query_ids"]

    def training_step(self, batch, batch_idx):
        loss, query_vec, query_id = self._step(batch)
        loss = torch.mean(loss)

        tensorboard_logs = {"train_loss": loss}
        self.log("train_loss", loss)
        return {"loss": loss, "query_vec": query_vec, "id": query_id}

    def on_validation_epoch_start(self):
        if self.local_rank == 0 and not self.hparams.dense_model and os.path.exists(self.hparams.last_vec_path_dev):
            os.remove(self.hparams.last_vec_path_dev)

    def validation_step(self, batch, batch_idx):
        input_ids,masks = [],[]

        input_ids,masks,scores = [],[],[]
        predicted_provance = []
        gold_question = []
        retrieved_pas = []
        with torch.no_grad():
            if not self.hparams.dense_model:
                # When using sparse model, we retrieve when validation end
                # print('questions:',batch['question'],'query_ids:', batch["query_ids"])
                query_vec = self.generate_vec(batch['question'])
                lines = []
                for id, query_vec in zip(batch["query_ids"], query_vec):
                    # print('query_vec shape:',query_vec.shape)
                    sparse_idxs = torch.nonzero(query_vec)
                    # print('sparse idxs shape:', sparse_idxs.shape)
                    sparse_values = query_vec[sparse_idxs]
                    tuple_vec = (sparse_idxs.cpu().numpy(), sparse_values.cpu().numpy())
                    dict_sparse = sparse_vector_to_dict(
                        tuple_vec, self.vocab_id2token, self.hparams.quantization_factor, dummy_token=self.dpr_tokenizer.unk_token)
                    lines.append((id, dict_sparse_to_string(dict_sparse)))
                with open(self.hparams.last_vec_path_dev, "a+") as output_file:
                    fcntl.flock(output_file, fcntl.LOCK_EX)
                    for qid, sparse_dic in lines:
                        output_file.write(str(qid) + "\t" + sparse_dic + os.linesep)
                    fcntl.flock(output_file, fcntl.LOCK_UN)
                self.log("val_loss", 0)
                return {"val_loss": 0}
            else:
                provance, scores = self.retriever.run(batch["question"], batch["query_ids"])
                # This test the ground truth retrieve
                # ret = self.retriever.retriever.generate_context_vectors(gp, self.ctx_encoder)
                # provance, scores = self.retriever.run_from_question_vectors(ret)
                # provance, scores = self.retriever.run(gp, batch["query_ids"])

                for batch_index, (x, p) in enumerate(zip(batch["source_txt"], provance)):
                    tmp_pp = []
                    for y in p:
                        # retrieved_pas.append(y['wikipedia_title']+'.'+y['text'])
                        # retrieved_pas.append(y['text'])
                        tmp_pp.append({"wikipedia_title":y["wikipedia_title"], "wikipedia_id":y["wikipedia_id"]})
                    predicted_provance.append(tmp_pp)

            
                self.log("val_loss", 0)
        return {'predicted_provance':predicted_provance, "val_loss": 0, 'ids': batch['query_ids'], 'question': batch['question']}
        # return {'predicted_provance':predicted_provance, "val_loss": loss, 'ids': batch['query_ids']}

    def on_train_epoch_start(self):
        self.my_epoch_count += 1
        # if self.hparams.reload_dataloaders_every_n_epochs == 0:
        if self.local_rank == 0 and os.path.exists(self.hparams.last_vec_path) and self.my_epoch_count:
            negative_samples_dir = self.hparams.negative_samples_dir + self.hparams.dataset + '-train-kilt.jsonl'
            if self.hparams.dense_model:
                if self.hparams.dis_faiss:
                    self.retriever.client_index.client.async_mode(True)
                with open(self.hparams.last_vec_path) as f:
                    lines = f.readlines()
                print(f"Load Saved Query Vectors From:{self.hparams.last_vec_path}")
                os.remove(self.hparams.last_vec_path)
                bz = self.hparams.eval_batch * 40
                predicted_provance = []
                query_ids = []
                for i in tqdm(range(0, len(lines), bz)):
                    datas = [json.loads(d)  for d in lines[i:min(i+bz, len(lines))]]
                    batch_question_vectors = []
                    # batch_query_ids = []
                    for d in datas:
                        batch_question_vectors.append(torch.tensor(d['query_vec']))
                        query_ids.append(d['id'])
                    self.eval()
                    with torch.no_grad():
                        provance, scores = self.retriever.run_from_question_vectors(torch.stack(batch_question_vectors))
                    self.train()

                    for p in provance:
                        tmp_pp = []
                        for y in p:
                            tmp_pp.append({"wikipedia_title":y["wikipedia_title"], "wikipedia_id":y["wikipedia_id"], 'text':y['text']})
                        predicted_provance.append(tmp_pp)
                if os.path.exists(negative_samples_dir):
                    os.remove(negative_samples_dir)
                with open(negative_samples_dir, 'w') as output_file:
                    data = []
                    for q_id, p in zip(query_ids, predicted_provance):
                        qa = {"id": q_id, 'input': "", 'output': []}
                        a = {'answer': "", 'provenance': p}
                        qa['output'].append(a)
                        data.append(json.dumps(qa))
                    # fcntl.flock(output_file, fcntl.LOCK_EX)
                    if os.stat(negative_samples_dir).st_size > 0:
                        output_file.write('\n')
                    output_file.write('\n'.join(data))
                if self.hparams.dis_faiss:
                    self.retriever.client_index.client.async_mode(False)
            else:
                os.system(
                    f"{self.hparams.anserini_path}/target/appassembler/bin/SearchCollection -hits {int(self.hparams.hits)} -parallelism 90 \
                                -index {self.hparams.save_index_dir} \
                                -topicreader {self.hparams.topicreader} -topics {self.hparams.last_vec_path} \
                                -output {self.hparams.last_sparse_output} -format trec \
                                -impact -pretokenized"
                )
                lines = load_list_from_file(self.hparams.last_sparse_output)

                query_id2provance = {}
                for line_info in lines:
                    sp = line_info.split(' ')
                    if len(sp) == 6:
                        qid, _, pid, rank, score, _method = sp
                    else:
                        qid1, qid2, _, pid, rank, score, _method = sp
                        qid = " ".join([qid1, qid2])
                    if query_id2provance.get(qid) is None:
                        query_id2provance[qid] = []
                    t_passage = self.retriever.all_passages[pid]
                    wiki_id = self.retriever.KILT_mapping.get(t_passage[1])
                    if wiki_id is None:
                        wiki_id = 0
                        print("Cound not find title in passage:", t_passage[1])

                    query_id2provance[qid].append(
                        {
                            "wikipedia_title": t_passage[1],
                            "wikipedia_id": wiki_id,
                            "text": t_passage[0],
                        }
                    )

                with open(negative_samples_dir, 'w') as output_file:
                    data = []
                    for q_id, p in query_id2provance.items():
                        qa = {"id": q_id, 'input': "", 'output': []}
                        a = {'answer': "", 'provenance': p}
                        qa['output'].append(a)
                        data.append(json.dumps(qa))
                    if os.stat(negative_samples_dir).st_size > 0:
                        output_file.write('\n')
                    output_file.write('\n'.join(data))
                os.remove(self.hparams.last_vec_path)
        # We wait main thread process the negative samples and then reload samples manually
        self.barrier()
        self.trainer.reset_train_dataloader(self)


    def get_line_count(self, filename):
        with open(filename) as f:
            return sum(1 for _ in f)

    def training_epoch_end(self, outputs):
        ids = []
        query_vec = []
        data = []
        # We prepare negative samples before the epoch really use them
        if (self.my_epoch_count + 1) % 2 == 0:
            if self.hparams.dense_model:
                for batch in outputs:
                    ids.extend(batch["id"])
                    query_vec.extend([i for i in batch["query_vec"]])
                for id, vec in zip(ids, query_vec):
                    data.append(json.dumps({'id':id, 'query_vec': vec.tolist()}))
                
                with open(self.hparams.last_vec_path, "a+") as output_file:
                    fcntl.flock(output_file, fcntl.LOCK_EX)
                    if os.stat(self.hparams.last_vec_path).st_size > 0:
                        output_file.write('\n')
                    output_file.write('\n'.join(data))
                    fcntl.flock(output_file, fcntl.LOCK_UN)
            else:
                lines = []
                for batch in outputs:
                    for id, query_vec in zip(batch["id"], batch["query_vec"]):
                        query_vec = query_vec.squeeze()
                        sparse_idxs = torch.nonzero(query_vec)
                        sparse_values = query_vec[sparse_idxs]
                        tuple_vec = (sparse_idxs.cpu().numpy(), sparse_values.cpu().numpy())
                        dict_sparse = sparse_vector_to_dict(
                            tuple_vec, self.vocab_id2token, self.hparams.quantization_factor, dummy_token=self.dpr_tokenizer.unk_token)
                        lines.append((id, dict_sparse_to_string(dict_sparse)))
                with open(self.hparams.last_vec_path, "a+") as output_file:
                    fcntl.flock(output_file, fcntl.LOCK_EX)
                    for qid, sparse_dic in lines:
                        output_file.write(str(qid) + "\t" + sparse_dic + os.linesep)
                    fcntl.flock(output_file, fcntl.LOCK_UN)
        return None


    def validation_epoch_end(self, outputs):
        # Direct set hits == 100
        if not self.hparams.dense_model:
            if self.local_rank == 0:
                commend = f"{self.hparams.anserini_path}/target/appassembler/bin/SearchCollection -hits {int(self.hparams.hits)} -parallelism 75 \
                                    -index {self.hparams.save_index_dir} \
                                    -topicreader {self.hparams.topicreader} -topics {self.hparams.last_vec_path_dev} \
                                    -output {self.hparams.last_sparse_output_dev} -format trec \
                                    -impact -pretokenized"
                os.system(commend)
                lines = load_list_from_file(self.hparams.last_sparse_output_dev)
                # with open(self.hparams.last_sparse_output_dev) as f:
                #     lines = load_list_from_file(f)
                query_id2provance = {}
                for line_info in lines:
                    sp = line_info.split(' ')
                    if len(sp) == 6:
                        qid, _, pid, rank, score, _method = sp
                    else:
                        qid1, qid2, _, pid, rank, score, _method = sp
                        qid = " ".join([qid1, qid2])
                    if query_id2provance.get(qid) is None:
                        query_id2provance[qid] = []
                    t_passage = self.retriever.all_passages[pid]
                    wiki_id = self.retriever.KILT_mapping.get(t_passage[1])
                    if wiki_id is None:
                        wiki_id = 0
                        print("Cound not find title in passage:", t_passage[1])

                    query_id2provance[qid].append(
                        {
                            "wikipedia_title": t_passage[1],
                            "wikipedia_id": wiki_id,
                            "text": t_passage[0],
                        }
                    )

                preds = []
                ids = []
                sources = []
                targets, provance = [], []
                for id, p in query_id2provance.items():
                    ids.append(id)
                    provance.append(p)
                sources = [''] * len(provance)
                preds = [''] * len(provance)
                seq2seq_to_kilt(ids, sources, preds, self.hparams.output_dir,
                                self.hparams.dataset + f"-{self.epoch_count}-sparse", 'dev', provance)
        else:
            preds = []
            ids = []
            sources = []
            targets, provance = [], []
            for batch in outputs:
                ids.extend(batch["ids"])
                provance.extend(batch["predicted_provance"])
                sources.extend(batch["question"])

            # sources = [''] * len(provance)
            preds = [''] * len(provance)
            seq2seq_to_kilt(ids, sources, preds, self.hparams.output_dir,
                            self.hparams.dataset + f"-{self.epoch_count}", 'dev', provance)
        tensorboard_logs = {}
        em = 0
        self.barrier()
        return {"avg_val_loss": 0, "log": tensorboard_logs, "EM": em}

    # Mostly come from RAG, using token decoding
    def marginalize(self, seq_logits, doc_scores):
        n_docs = self.hparams.n_docs

        # RAG-token marginalization
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )

        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)

        return torch.logsumexp(log_prob_sum, dim=1)

    # Mostly come from RAG
    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0):
        n_docs = self.hparams.n_docs
        # shift tokens left
        # target = torch.cat(
        #     [target[:, 1:], target.new(target.shape[0], 1).fill_(self.tokenizer.pad_token_id)], 1
        # )

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.tokenizer.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        rag_logprobs = self.marginalize(seq_logits, doc_scores)

        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()
        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)  # sum over tokens
        smooth_obj = smooth_obj.sum(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    def extend_source(self, source, title, passage):
        ret_source = []
        for t,p in zip(title, passage):
            if t.startswith('"'):
                t = t[1:]
            if t.endswith('"'):
                t = t[:-1]
            tp_part = t + '[SEP]' + p

            tp_part_ids = self.tokenizer.encode(tp_part, add_special_tokens=False)[:self.source_length]

            ret_source.append(self.tokenizer.decode(source[:100] +  tp_part_ids + [self.tokenizer.sep_token_id] + source[100:], add_special_tokens = False))

        tokenized = self.tokenizer.batch_encode_plus(
                ret_source, add_special_tokens=True, max_length=self.hparams.max_sequence_length, pad_to_max_length=True,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
                )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def remove_padding(self, x):
        while x[-1] == self.tokenizer.pad_token_id:
            x.pop()
        return x
        # return self.tokenizer.decode(x)

    def collate_fn(self, batch):

        # Retrieve source document
        query_data = []


        for element in batch:
            query_data.append(
                {"query": element["query_txt"], "id": element["id"]}
            )

        n_docs = self.hparams.n_docs
        source_txt = [self.remove_padding(x["source_ids"]) for x in batch]

        # question, query_ids =  self.retriever.feed_data(query_data)
        question = [x["query"] for x in query_data]
        query_ids = [x["id"] for x in query_data]
        target_ids = torch.stack([x["target_ids"] for x in batch])
        ret = {"question": question, "query_ids": query_ids, "source_txt": source_txt, "target_ids": target_ids}
        if batch[0].get('gold_provenance'):
            ret['gold_provenance'] = [x['gold_provenance'] for x in batch]
        if batch[0].get("negative_samples"):
            ret["negative_samples"]  = [x["negative_samples"] for x in batch]
        return ret

        # return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y, "ids": ids, "scores": scores, "query_data":query_data}

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        random.seed(self.my_epoch_count)
        datasets = []
        for d in self.dataset_list:
            datasets.append(
                KiltDataset(self.tokenizer, self.data_dir, d, type_path, self.source_length, self.target_length,
                            self.output_dir, self.hparams))
        if type_path == 'dev':
            for x in datasets:
                self.devsets.update(x.id_targets)
        concat_dataset = ConcatDataset(datasets)
        dataloader = DataLoader(concat_dataset, num_workers=8, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, drop_last=True)

        print(type_path, dataloader.batch_size, concat_dataset.__len__())
        return dataloader

    def get_test_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        datasets = []
        for d in self.dataset_list:
            datasets.append(
                KiltDataset(self.tokenizer, self.data_dir, d, type_path, self.source_length, self.target_length,
                            self.output_dir, self.hparams))
        # if type_path == 'dev':
        for x in datasets:
            self.devsets.update(x.id_targets)
        concat_dataset = ConcatDataset(datasets)
        dataloader = DataLoader(concat_dataset, num_workers=8, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, drop_last=False)

        print(type_path, dataloader.batch_size, concat_dataset.__len__())
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.train_batch_size, shuffle=False)
        t_total = (
                (len(dataloader.dataset) // (self.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler

        return dataloader
    
    def ng_gen_dataloader(self) -> DataLoader:
        dataloader = self.get_test_dataloader("train", batch_size=self.train_batch_size, shuffle=False)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        # return self.get_dataloader("train", batch_size=self.eval_batch_size)
        return self.get_test_dataloader("dev", batch_size=self.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_test_dataloader("test_without_answers", batch_size=self.eval_batch_size)
        # return self.get_dataloader("test", batch_size=self.eval_batch_size)
    @staticmethod
    def add_model_specific_args(arg_parser, root_dir):
        BaseTransformer.add_model_specific_args(arg_parser, root_dir)

        arg_parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the dataset files for the task.",
        )
        arg_parser.add_argument("--topk",default=1,type=int,required=True,help="Set up the how many passages are recalled.")
        arg_parser.add_argument("--dataset", required=True, type=str)

        return arg_parser
        
@hydra.main(config_path="conf", config_name="biencoder_train_cfg")
def main(arguments):
    # If true, this will cause deadlock sometime !
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    # If output_dir not provided, a folder will be generated in pwd
    if not arguments.output_dir:
        arguments.output_dir = os.path.join("./results", f"{arguments.task}_{time.strftime('%Y%m%d_%H%M%S')}", )
        os.makedirs(arguments.output_dir)

    model = Seq2seqTransformer(arguments)
    # load configs
    with open(arguments.test_config, "r") as fin:
        test_config_json = json.load(fin)
    # create a new directory to log and store results
    log_directory = utils.create_logdir_with_timestamp(arguments.logdir)
    logger = None

    logger = utils.init_logging(log_directory, arguments.model_name, logger)
    logger.info("loading {} ...".format(arguments.model_name))

    trainer = generic_train(model, arguments)

    if arguments.do_predict:
        checkpoints = list(
            sorted(glob.glob(os.path.join(arguments.output_dir, "*.ckpt"), recursive=True)))
        try:
            model = model.load_from_checkpoint(checkpoints[-1])
            model.hparams.dataset = arguments.dataset
            model.dataset_list = arguments.dataset.split(',')
            print("Model Load Done!")
        except:
            print("Skip predicting due to error happened")
        trainer.test(model)


if __name__ == "__main__":
    main()
