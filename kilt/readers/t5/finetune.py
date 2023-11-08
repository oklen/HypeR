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
from transformers import PreTrainedModel
#from transformers.tokenization_utils import trim_batch

from base_transformer import BaseTransformer, add_generic_args, generic_train
from data import KiltDataset, seq2seq_to_kilt, dataset_config
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
                               'Fact Checking:', 'Dialogue:', 'Relation Extraction:', '[SEP]'])  #
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': special_tokens})

        fevers_classes = ["<SUPPORTS>", "<REFUTES>"]

        self.tokenizer.add_tokens(fevers_classes)

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.n_positions = hparams.max_sequence_length

        self.bad_words = [[self.tokenizer.convert_tokens_to_ids(bad_word)] for bad_word in
                          self.tokenizer.additional_special_tokens]
        self.eos_token = self.tokenizer.eos_token
        
        for d in self.dataset_list:
            # train_batch = int(hparams.datasets[d]['train_batch'])
            # eval_batch = int(hparams.datasets[d]['eval_batch'])
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
        # self.retriever = None

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        # return self.model(
        #     input_ids, attention_mask=attention_mask, lm_labels=lm_labels,
        # )

        return self.model(
            input_ids, attention_mask=attention_mask, labels=lm_labels,
        )

    def _step(self, batch):
        input_ids,masks = [],[]

        # for x in batch["source_txt"]:
        #     tmp_input_ids, tmp_masks = self.extend_source(x, ["Go"], ["Go"])
        #     input_ids.extend(tmp_input_ids)
        #     masks.extend(tmp_masks)
        # input_ids = torch.stack(input_ids)
        # masks = torch.stack(masks)
        # scores = torch.stack([torch.tensor(0.5) for x in batch["target_ids"]]).unsqueeze(-1)

        input_ids,masks,scores = [],[],[]
        provance, question_vectors = self.retriever.run(batch["question"], batch["query_ids"])
        for batch_index, (x, tmp_id) in enumerate(zip(batch["source_txt"], batch["query_ids"])):
            # for p in provance[tmp_id]:
            p = provance[tmp_id]
            tmp_input_ids, tmp_masks = self.extend_source(x, [y["wikipedia_title"] for y in p], 
            [y["text"] for y in p])
            # scores.append(torch.stack([torch.tensor(v["vector"]).to(question_vectors) * question_vectors[batch_index] for v in p]))
            scores.append(torch.stack([torch.sum(torch.tensor(v["vector"]).to(question_vectors) * question_vectors[batch_index], -1) for v in p]))
            input_ids.extend(tmp_input_ids)
            masks.extend(tmp_masks)
        
        source_ids = torch.stack(input_ids)
        source_mask = torch.stack(masks)
        scores = torch.stack(scores)

        n_docs = self.hparams.n_docs

        # masks = torch.stack([x["source_mask"] for x in batch for y in range(n_docs)])
        target_ids = torch.stack([x for x in batch["target_ids"]])

        # ids = [x["id"] for x in batch]

        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(source_ids, pad_token_id, attention_mask=source_mask)

        lm_labels = y.clone()
        lm_labels[y == pad_token_id] = -100
        lm_labels = lm_labels.unsqueeze(1).repeat(1, n_docs, 1).reshape(-1, lm_labels.size(-1))


        outputs = self(source_ids.cuda(), attention_mask=source_mask.cuda(), lm_labels=lm_labels.cuda(), )
        loss = self.get_token_decoding_nll(outputs.logits, scores.cuda(), y.cuda(), epsilon=0.01)
        # loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # input_ids, masks = torch.stack([x["source_ids"] for x in batch])
        input_ids,masks,scores = [],[],[]
        provance, question_vectors = self.retriever.run(batch["question"], batch["query_ids"])
        for batch_index, (x, tmp_id) in enumerate(zip(batch["source_txt"], batch["query_ids"])):
            # for p in provance[tmp_id]:
            p = provance[tmp_id]
            tmp_input_ids, tmp_masks = self.extend_source(x, [y["wikipedia_title"] for y in p], 
            [y["text"] for y in p])
            # scores.append(torch.stack([torch.tensor(v["vector"]).to(question_vectors) * question_vectors[batch_index] for v in p]))
            scores.append(torch.stack([torch.sum(torch.tensor(v["vector"]).to(question_vectors) * question_vectors[batch_index], -1) for v in p]))
            input_ids.extend(tmp_input_ids)
            masks.extend(tmp_masks)
        
        source_ids = torch.stack(input_ids)
        source_mask = torch.stack(masks)
        scores = torch.stack(scores)

        # input_ids,masks = torch.stack([y for x in batch["source_txt"] for y in self.extend_source(x, "Go", \
        #     "Go")])

        # scores = torch.stack([torch.tensor(0.5)]).unsqueeze(-1)

        # input_ids = torch.stack(y for x in batch for y in self.extend_source(x["source_txt"], provance[x["id"]["title"]], \
        #     provance[x["id"]["passage"]]))
        # scores = torch.stack([provance[x["id"]]["score"]] for x in batch).cuda()

        n_docs = self.hparams.n_docs
        # masks = torch.stack([x for x in batch["source_mask"] for y in range(n_docs)])
        target_ids = torch.stack([x for x in batch["target_ids"]])

        # ids = [x["id"] for x in batch]
        pad_token_id = self.tokenizer.pad_token_id
        # source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        
        generated_ids = self.generate(
            input_ids=None,
            attention_mask=None,
            context_input_ids=source_ids.cuda(),
            context_attention_mask=source_mask.cuda(),
            doc_scores=scores
        )

        # source_ids, source_mask, y = KiltDataset.trim_seq2seq_batch(batch, pad_token_id)

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
        target = [self.tokenizer.decode(t) for t in target_ids]
        loss = self._step(batch)
        sources = [self.tokenizer.decode(s) for s in source_ids]

        self.log("val_loss", loss)

        return {"val_loss": loss, 'sources': sources, "preds": preds, "target": target, "ids": batch["query_ids"]}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

        preds = []
        ids = []
        sources = []
        for batch in outputs:
            sources.extend(batch['sources'])
            preds.extend(batch['preds'])
            ids.extend(batch["ids"])
        em = 0
        for q_id, pred in set(zip(ids, preds)):
            targets = [normalize_answer(x) for x in self.devsets[q_id]]

            if normalize_answer(pred) in targets:
                em = em + 1
        if em > self.em:
            self.em = em
            self.trainer.save_checkpoint(self.output_dir + '/' + "best_em.ckpt")
            seq2seq_to_kilt(set(ids), set(sources), set(preds), self.hparams.output_dir,
                            self.hparams.dataset, 'dev')
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, "EM": em}

    # Mostly come from RAG, using token decoding
    def token_marginalize(self, seq_logits, doc_scores):
        n_docs = self.hparams.n_docs

        # RAG-token marginalization
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    # Mostly come from RAG
    def get_token_decoding_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0):
        # n_docs = self.hparams.n_docs
        # shift tokens left
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.tokenizer.pad_token_id)], 1
        )

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.tokenizer.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        rag_logprobs = self.token_marginalize(seq_logits, doc_scores)

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
        input_ids,masks,scores = [],[],[]
        provance, question_vectors = self.retriever.run(batch["question"], batch["query_ids"])
        for batch_index, (x, tmp_id) in enumerate(zip(batch["source_txt"], batch["query_ids"])):
            # for p in provance[tmp_id]:
            p = provance[tmp_id]
            tmp_input_ids, tmp_masks = self.extend_source(x, [y["wikipedia_title"] for y in p], 
            [y["text"] for y in p])
            scores.append(torch.stack([torch.sum(torch.tensor(v["vector"]).to(question_vectors) * question_vectors[batch_index], -1) for v in p]))
            input_ids.extend(tmp_input_ids)
            masks.extend(tmp_masks)
        
        source_ids = torch.stack(input_ids)
        source_mask = torch.stack(masks)
        scores = torch.stack(scores)
        pad_token_id = self.tokenizer.pad_token_id

        # source_ids, source_mask, y = KiltDataset.trim_seq2seq_batch(batch, pad_token_id)
        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_kilt_task.py

        generated_ids = self.generate(
            input_ids=None,
            attention_mask=None,
            context_input_ids=source_ids,
            context_attention_mask=source_mask,
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
        loss = self._step(batch)
        sources = [self.tokenizer.decode(s) for s in source_ids]
        return {"val_loss": loss, 'sources': sources, "preds": preds, "target": target, "ids": batch["ids"]}

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
        for batch in outputs:
            sources.extend(batch['sources'])
            preds.extend(batch['preds'])
            ids.extend(batch["ids"])
        
        seq2seq_to_kilt(ids, sources, preds, self.hparams.output_dir, self.hparams.dataset, 'predict_test')

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def extend_source(self, source, title, passage):
        ret_source = []
        for t,p in zip(title, passage):
            if t.startswith('"'):
                t = t[1:]
            if t.endswith('"'):
                t = t[:-1]
            ret_source.append(t + self.eos_token + p + self.eos_token + source)

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
        return self.tokenizer.decode(x)

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
        target_ids = [x["target_ids"] for x in batch]

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
        dataloader = DataLoader(concat_dataset, num_workers=0, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

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
        dataloader = DataLoader(concat_dataset, num_workers=0, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

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
        """
        Implements RAG token decoding.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (`int`, *optional*, defaults to 10):
                The minimum length of the sequence to be generated.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether or not to stop the beam search when at least `num_beams` sentences are finished per batch or
                not.
            use_cache: (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
                `decoder_input_ids`.
            bad_words_ids(`List[int]`, *optional*):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            num_beam_groups (`int`, *optional*, defaults to 1):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the `generator`'s `[`~generation_utils.GenerationMixin.generate`] function,
                where we set `num_return_sequences` to `num_beams`. decoder_start_token_id (`int`, *optional*): If an
                encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments `inputs_ids` and the batch ID
                `batch_id`. It has to return a list with the allowed tokens for the next generation step conditioned on
                the previously generated tokens `inputs_ids` and the batch ID `batch_id`. This argument is useful for
                constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                 Custom logits processors that complement the default logits processors built from arguments and a
                 model's config. If a logit processor is passed that is already created with the arguments or a model's
                 config an error is thrown.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
                 model's config. If a stopping criteria is passed that is already created with the arguments or a
                 model's config an error is thrown.
            forced_bos_token_id (`int`, *optional*):
                The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
                for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
                the target language token.
            forced_eos_token_id (`int`, *optional*):
                The id of the token to force as the last generated token when `max_length` is reached.
            remove_invalid_values (`bool`, *optional*):
                Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
                crash. Note that using `remove_invalid_values` can slow down generation.

        Return:
            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches
            finished early due to the `eos_token_id`.
        """
        # set default parameters
        n_docs = n_docs if n_docs is not None else self.hparams.n_docs
        num_beams = num_beams if num_beams is not None else self.hparams.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.hparams.num_beam_groups
        max_length = max_length if max_length is not None else self.hparams.max_length
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.hparams.num_return_sequences
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.tokenizer.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        use_cache = use_cache if use_cache is not None else self.hparams.use_cache

        # T5 use pad token as the starting token for decoder_input_ids generation
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.tokenizer.pad_token_id
        )

        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.hparams.remove_invalid_values
        )

        exponential_decay_length_penalty = (
            exponential_decay_length_penalty
            if exponential_decay_length_penalty is not None
            else self.hparams.exponential_decay_length_penalty
        )

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

        input_ids = torch.full(
            (batch_size * num_beams * n_docs, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        input_ids_seq_length = input_ids.shape[-1]
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


        pre_processor = self.model._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            # input_ids_seq_length=input_ids_seq_length,
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
            # exponential_decay_length_penalty=exponential_decay_length_penalty,
            # logits_processor=logits_processor,
            # renormalize_logits=renormalize_logits,
        )

        if num_beams == 1:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            return self.model.greedy_search(
                input_ids,
                logits_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        elif num_beams > 1:
            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams * n_docs,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            
            return self.model.beam_search(
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
        model = model.load_from_checkpoint(checkpoints[-1])
        model.hparams.dataset = arguments.dataset
        model.dataset_list = arguments.dataset.split(',')

        trainer.test(model)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # add_generic_args(parser)
    # parser = Seq2seqTransformer.add_model_specific_args(parser, os.getcwd())
    # args = parser.parse_args()

    main()
