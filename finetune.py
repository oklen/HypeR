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

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import PreTrainedModel, T5Model
from transformers.utils import ModelOutput
#from transformers.tokenization_utils import trim_batch

from base_transformer import BaseTransformer, add_generic_args, generic_train
from data import KiltDataset, seq2seq_to_kilt, dataset_config, remove_preds
from kilt.eval_downstream import normalize_answer
# from kilt.kilt_utils import utils
import kilt.kilt_utils as utils
from typing import Callable, List, Optional, Tuple, Union

from kilt.retrievers import DPR_connector

from transformers import LogitsProcessorList, BeamSearchScorer, PretrainedConfig,StoppingCriteriaList
# from ...modeling_outputs import ModelOutput
# from ...modeling_utils import PreTrainedModel



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
 

        special_tokens = []

        for i in range(0, 101):
            special_tokens.append('<extra_id_' + str(i) + '>')

        special_tokens.extend(['[START_ENT]', '[END_ENT]', 'Question Answering:', 'Entity Linking:',
                               'Fact Checking:', 'Dialogue:', 'Relation Extraction:', '[SEP]'])
        self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        
        # self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)


        fevers_classes = ["<SUPPORTS>", "<REFUTES>"]

        self.tokenizer.add_tokens(fevers_classes, special_tokens=True)

        self.model.resize_token_embeddings(len(self.tokenizer))


        last_length = len(self.tokenizer)
        for param in self.model.parameters():
            param.requires_grad = False

        #Add Soft-Prompt
        if hparams.use_prompt:
            Task_type = ['Question Answering', 'Entity Linking', 'Fact Checking', 'Dialogue', 'Relation Extraction','Slot Filling']
            prompt_tokens = []
            for t in Task_type:
                for i in range(100):
                    prompt_tokens.append(t+'#'+str(i))
            
            self.tokenizer.add_tokens(prompt_tokens, special_tokens=True)

            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.set_input_embeddings(DPR_connector.SoftEmbedding(self.model.get_input_embeddings(), hparams.soft_prompt_length))


        self.model.config.n_positions = hparams.max_sequence_length

        self.bad_words = [[self.tokenizer.convert_tokens_to_ids(bad_word)] for bad_word in
                          self.tokenizer.additional_special_tokens]
        self.eos_token = self.tokenizer.eos_token
        
        #print(dataset_config.keys())
        #print(dict(dataset_config))
        for d in self.dataset_list:
            train_batch = int(hparams['train_batch'])
            eval_batch = int(hparams['eval_batch'])
            source_length = int(hparams.datasets[d]['source_length'])
            target_length = int(hparams.datasets[d]['target_length'])
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
        if not hparams.debug_mode:
            self.question_encoder = self.retriever.encoder # register question decoder there
        
        self.retriever.client_index.client.set_batch_size(self.hparams.n_gpu)

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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        do_marginalize: Optional[bool] = None,
        reduce_loss: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        n_docs: Optional[int] = None,
        y = None,
        **kwargs  # needs kwargs for generation
    ):
        n_docs = n_docs if n_docs is not None else self.hparams.n_docs
        # do_marginalize = do_marginalize if do_marginalize is not None else self.hparams.do_marginalize
        do_marginalize = True
        reduce_loss = reduce_loss if reduce_loss is not None else self.hparams.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        # if decoder_input_ids is not None:
        #     decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)

        # if decoder_attention_mask is not None:
        #     decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)

        if input_ids is not None:
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = None
            logits = outputs.logits
            if labels is not None:
                assert decoder_input_ids is not None
                loss = self.get_nll(
                    outputs.logits,
                    doc_scores,
                    y,
                    # labels,
                    reduce_loss=reduce_loss,
                    epsilon=self.hparams.label_smoothing,
                    # n_docs=n_docs,
                )
            # loss = outputs.loss
        else:
            outputs = self.model(
                # input_ids=input_ids,
                # input_ids=input_ids if input_ids is not None else context_input_ids,
                # attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                # context_input_ids=context_input_ids,
                # context_attention_mask=context_attention_mask,
                # doc_scores=doc_scores,
                past_key_values=past_key_values,
                use_cache=use_cache,
                # use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                # output_retrieved=output_retrieved,
                # n_docs=n_docs,
            )
            logits = outputs.logits
            loss = outputs.loss
            if do_marginalize and doc_scores is not None:
                logits = self.marginalize(logits, doc_scores)
            



        return ModelOutput(
            loss=loss,
            logits=logits,
            doc_scores=doc_scores,
            past_key_values=outputs.past_key_values,
            # context_input_ids=outputs.context_input_ids,
            # context_attention_mask=outputs.context_attention_mask,
            # retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            # retrieved_doc_ids=outputs.retrieved_doc_ids,
            # question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            # question_enc_hidden_states=outputs.question_enc_hidden_states,
            # question_enc_attentions=outputs.question_enc_attentions,
            # generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            # generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            # generator_enc_attentions=outputs.generator_enc_attentions,
            # generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            # generator_dec_attentions=outputs.generator_dec_attentions,
            # generator_cross_attentions=outputs.generator_cross_attentions,
        )
    # def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
    #     # print(self.tokenizer.decode(input_ids[0]))
    #     # print(input_ids.shape)
    #     res =  self.model(
    #         input_ids, attention_mask=attention_mask, labels=lm_labels,
    #     )
    #     return res.logits

    def _step(self, batch):
        input_ids,masks = [],[]

        input_ids,masks,scores = [],[],[]
        provance, scores = self.retriever.run(batch["question"], batch["query_ids"])

        for batch_index, (x, p) in enumerate(zip(batch["source_txt"], provance)):

            tmp_input_ids, tmp_masks = self.extend_source(x, [y["wikipedia_title"] for y in p], [y["text"] for y in p])
            # scores.append(torch.stack([torch.sum(torch.tensor(v["vector"]).to(question_vectors) * question_vectors[batch_index], -1) for v in p]))
            # scores.append(torch.stack([x['score'] for x in provance[tmp_id]]))
            input_ids.extend(tmp_input_ids)
            masks.extend(tmp_masks)
        
        source_ids = torch.stack(input_ids).cuda()
        source_mask = torch.stack(masks).cuda()
        target_ids = batch["target_ids"].cuda()
        # scores = torch.stack(scores)
        # print("source_ids shape:", source_ids.shape)

        n_docs = self.hparams.n_docs

        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(source_ids, pad_token_id, attention_mask=source_mask)

        lm_labels = y.clone()
        lm_labels[y == pad_token_id] = -100
        lm_labels = lm_labels.unsqueeze(1).repeat(1, n_docs, 1).reshape(-1, lm_labels.size(-1))
        # print("lm_labels:", lm_labels.shape)

        outputs = self(source_ids, attention_mask=source_mask, labels=lm_labels, doc_scores=scores, y=y)
        # outputs = self(source_ids, attention_mask=source_mask, labels=lm_labels)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = torch.mean(self._step(batch))

        tensorboard_logs = {"train_loss": loss}
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids,masks = [],[]

        input_ids,masks,scores = [],[],[]
        predicted_provance = []
        provance, scores = self.retriever.run(batch["question"], batch["query_ids"])
        for batch_index, (x, p) in enumerate(zip(batch["source_txt"], provance)):
            tmp_title, tmp_text = [], []
            tmp_pp = []
            for y in p:
                tmp_title.append(y['wikipedia_title'])
                tmp_text.append(y['text'])
                tmp_pp.append({"wikipedia_title":y["wikipedia_title"], "wikipedia_id":y["wikipedia_id"]})

            predicted_provance.append(tmp_pp)

            tmp_input_ids, tmp_masks = self.extend_source(x, tmp_title, tmp_text)
            # scores.append(torch.stack([torch.sum(torch.tensor(v["vector"]).to(question_vectors) * question_vectors[batch_index], -1) for v in p]))
            # scores.append(torch.stack([x['score'] for x in provance[tmp_id]]))

            input_ids.extend(tmp_input_ids)
            masks.extend(tmp_masks)
        
        source_ids = torch.stack(input_ids).cuda()
        source_mask = torch.stack(masks).cuda()
        # print(batch["target_ids"])
        # print(self.tokenizer.decode(batch["target_ids"][0]))
        target_ids = batch["target_ids"].cuda()

        n_docs = self.hparams.n_docs

        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(source_ids, pad_token_id, attention_mask=source_mask)

        # preds = [self.tokenizer.decode(g) for g in source_ids]
        # print("Inputs:", preds[0])
        # print("input:",preds[0])

        generated_ids = self.generate(
            input_ids=None,
            attention_mask=None,
            context_input_ids=source_ids.cuda(),
            context_attention_mask=source_mask.cuda(),
            doc_scores=scores
        )
        preds = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        target = [self.tokenizer.decode(t) for t in y]
        # print("preds:", preds[0])
        # print("target:", target[0])
        # exit(0)

        lm_labels = y.clone()
        lm_labels[y == pad_token_id] = -100
        lm_labels = lm_labels.unsqueeze(1).repeat(1, n_docs, 1).reshape(-1, lm_labels.size(-1))

        # outputs = self(source_ids, attention_mask=source_mask, labels=lm_labels, doc_scores=scores, )
        outputs = self.model(source_ids, attention_mask=source_mask, labels=lm_labels)
        loss = outputs.loss.cpu()
        # loss = 0
        # loss = self.get_nll(logits, scores, y, epsilon=0.01).cpu()

        sources = [self.tokenizer.decode(s) for s in source_ids]

        self.log("val_loss", loss)

        return {"val_loss": loss, 'sources': sources, "preds": preds, "target": target, "ids": batch["query_ids"], 'predicted_provance':predicted_provance}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        preds = []
        ids = []
        sources = []
        targets,provance = [],[]
        for batch in outputs:
            sources.extend(batch['sources'])
            preds.extend(batch['preds'])
            ids.extend(batch["ids"])
            targets.extend(batch["target"])
            provance.extend(batch["predicted_provance"])

        em = 0

        for q_id, pred, target in zip(ids, preds, targets):
            targets = [normalize_answer(x) for x in self.devsets[q_id]]
            print("q_id:", q_id)
            print("nonr-answer-preds:", pred)
            print("preds:", normalize_answer(pred))
            print("trained target:", target)
            print("match targets:", targets)
            if normalize_answer(pred) in targets:
                em = em + 1
        print("EM:",em)

        if em > self.em:
            self.em = em
            self.trainer.save_checkpoint(self.output_dir + '/' + "best_em.ckpt")
        seq2seq_to_kilt(ids, sources, preds, self.hparams.output_dir,
                        self.hparams.dataset + f"-{self.epoch_count}", 'dev', provance)
            # seq2seq_to_kilt(set(ids), set(sources), set(preds), self.hparams.output_dir,
            #                 self.hparams.dataset, 'dev')
        tensorboard_logs = {"val_loss": avg_loss, "EM": em}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, "EM": em}

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

    def test_step(self, batch, batch_idx):
        input_ids,masks = [],[]

        input_ids,masks,scores = [],[],[]
        provance, scores = self.retriever.run(batch["question"], batch["query_ids"])
        predicted_provance = []
        top_score_index = torch.max(scores, -1).indices
        for batch_index, (x, p) in enumerate(zip(batch["source_txt"], provance)):
            title_one_batch = []
            text_one_batch = []
            for y in p:
                title_one_batch.append(y["wikipedia_title"])
                text_one_batch.append(y["text"])
            top_score_id = top_score_index[batch_index]
            predicted_provance.append({"wikipedia_title":p[top_score_id]["wikipedia_title"],
            "wikipedia_id":p[top_score_id]["wikipedia_id"]})
            wiki_ids_one_batch = [y["wikipedia_id"] for y in p]
            tmp_input_ids, tmp_masks = self.extend_source(x, title_one_batch, text_one_batch)

            input_ids.extend(tmp_input_ids)
            masks.extend(tmp_masks)
        
        source_ids = torch.stack(input_ids).cuda()
        source_mask = torch.stack(masks).cuda()
        target_ids = batch["target_ids"].cuda()

        n_docs = self.hparams.n_docs
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(source_ids, pad_token_id, attention_mask=source_mask)

        lm_labels = y.clone()
        lm_labels[y == pad_token_id] = -100
        lm_labels = lm_labels.unsqueeze(1).repeat(1, n_docs, 1).reshape(-1, lm_labels.size(-1))

        
        generated_ids = self.generate(
            input_ids=None,
            attention_mask=None,
            context_input_ids=source_ids.cuda(),
            context_attention_mask=source_mask.cuda(),
            doc_scores=scores
        )
        
        # generated_ids = self.model.generate(
        #     input_ids=source_ids,
        #     attention_mask=source_mask,
        #     num_beams=1,
        #     max_length=self.target_length,
        #     repetition_penalty=1,
        #     length_penalty=1.0,
        #     early_stopping=True,
        #     use_cache=True,
        #     do_sample=False,
        #     top_p=0.95,
        #     top_k=50,
        #     bad_words_ids=self.bad_words
        # )
        preds = [self.tokenizer.decode(g) for g in generated_ids]
        target = [self.tokenizer.decode(t) for t in y]

        # outputs = self.model(source_ids, attention_mask=source_mask, labels=lm_labels)
        # loss = outputs.loss.cpu()
        # loss = self.get_nll(logits, scores, y, epsilon=0.01)
        sources = [self.tokenizer.decode(s) for s in source_ids]
        # return {"test_loss": loss, 'sources': sources, "preds": preds, "target": target, "ids": batch["query_ids"], "predicted_provance":predicted_provance}
        return {'sources': sources, "preds": preds, "target": target, "ids": batch["query_ids"], "predicted_provance":predicted_provance}

    # def test_end(self, outputs):
    #     sources = []
    #     preds = []
    #     ids = []
    #     for batch in outputs:
    #         sources.extend(batch['sources'])
    #         preds.extend(batch['preds'])
    #         ids.extend(batch["ids"])
        
    #     seq2seq_to_kilt(ids, sources, preds, self.hparams.output_dir, self.hparams.dataset, 'test')

    #     return self.test_epoch_end(outputs)

    def test_epoch_end(self, outputs):
        sources = []
        preds = []
        ids = []
        provance = []
        ans = []
        for batch in outputs:
            sources.extend(batch['sources'])
            preds.extend(batch['preds'])
            ids.extend(batch["ids"])
            provance.extend(batch["predicted_provance"])
            ans.extend(batch["target"])
        
        seq2seq_to_kilt(ids, sources, preds, self.hparams.output_dir, self.hparams.dataset, 'predict_test', provance)
        seq2seq_to_kilt(ids, sources, ans, self.hparams.output_dir, self.hparams.dataset, 'ans_test', provance)

        # avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        # tensorboard_logs = {"test_loss": avg_loss}
        # return {"avg_val_loss": avg_loss, "log": tensorboard_logs}
        return {}

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

        return {"question": question, "query_ids": query_ids, "source_txt": source_txt, "target_ids": target_ids}

        # return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y, "ids": ids, "scores": scores, "query_data":query_data}

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        datasets = []
        for d in self.dataset_list:
            datasets.append(
                KiltDataset(self.tokenizer, self.data_dir, d, type_path, self.source_length, self.target_length,
                            self.output_dir))
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
                            self.output_dir))
        if type_path == 'dev':
            for x in datasets:
                self.devsets.update(x.id_targets)
        concat_dataset = ConcatDataset(datasets)
        dataloader = DataLoader(concat_dataset, num_workers=8, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, drop_last=False)

        print(type_path, dataloader.batch_size, concat_dataset.__len__())
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.train_batch_size, shuffle=True)
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

    def val_dataloader(self) -> DataLoader:
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
        
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        num_beams: Optional[int] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[List[List[int]]] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        n_docs: Optional[int] = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
        **model_kwargs
    ) -> torch.LongTensor:

        # set default parameters
        n_docs = n_docs if n_docs is not None else self.hparams.n_docs
        num_beams = num_beams if num_beams is not None else self.hparams.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.hparams.num_beam_groups
        max_length = max_length if max_length is not None else self.hparams.max_length
        min_length = min_length if min_length is not None else self.hparams.min_length
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.hparams.num_return_sequences
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.tokenizer.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        use_cache = use_cache if use_cache is not None else self.hparams.use_cache
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.hparams.repetition_penalty

        # T5 use pad token as the starting token for decoder_input_ids generation
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.tokenizer.pad_token_id
        )

        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.hparams.remove_invalid_values
        )

        # templaor work around
        # exponential_decay_length_penalty = (
        #     exponential_decay_length_penalty
        #     if exponential_decay_length_penalty is not None
        #     else eval(self.hparams.exponential_decay_length_penalty)
        # )

        # retrieve docs
        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            out = self.retriever(
                input_ids,
                question_hidden_states.cpu().detach().to(torch.float32).numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="pt",
            )
            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )

            # set to correct device
            retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states)
            context_input_ids = context_input_ids.to(input_ids)
            context_attention_mask = context_attention_mask.to(input_ids)

            # compute doc_scores
            doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(
                1
            )

        assert (
            context_input_ids.shape[0] % n_docs
        ) == 0, f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}."

        # batch_size
        batch_size = context_input_ids.shape[0] // n_docs

        encoder = self.model.get_encoder()
        encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=context_attention_mask, return_dict=True)

        last_hidden_state = encoder_outputs["last_hidden_state"]
        
        def extend_enc_output(tensor, num_beams=None):
            # split into `batch_size`, `num_beams`, `num_docs`
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            # repeat same last hidden states over `num_beams` dimension
            tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
            # merge `batch_size`, `num_beams`, `num_docs` dims again
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])

        
        # correctly extend last_hidden_state and attention mask
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(last_hidden_state, num_beams=num_beams)

        doc_scores = doc_scores.repeat_interleave(num_beams, dim=0)

        # define start_len & additional parameters
        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["n_docs"] = n_docs


        num_beams = num_beams * n_docs
        input_ids = torch.full(
            (batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        input_ids_seq_length = input_ids.shape[-1]

        pre_processor = self.model._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=context_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
        )

        if num_beams == 1:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            return self.greedy_search(
                input_ids,
                logits_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        elif num_beams > 1:
            try:
                length_penalty = length_penalty if length_penalty is not None else self.hparams.length_penalty
                early_stopping = early_stopping if early_stopping is not None else self.hparams.early_stopping
            except:
                length_penalty = None
                early_stopping = None
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                # num_beams=num_beams * n_docs,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        else:
            raise ValueError(f"`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {num_beams}")
import json

@hydra.main(config_path="conf", config_name="biencoder_train_cfg")
def main(arguments):
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
    print(trainer.test(model))


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # add_generic_args(parser)
    # parser = Seq2seqTransformer.add_model_specific_args(parser, os.getcwd())
    # args = parser.parse_args()

    main()
