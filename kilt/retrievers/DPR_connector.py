# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import json
import argparse
import glob
import pickle
import torch
import os
import joblib
import gc
import time
import math
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import Optional,Tuple,Union
from pytorch_lightning.utilities.distributed import gather_all_tensors

from kilt.kilt_utils import *
logger = init_logging('./','T5')

from dpr.utils.model_utils import (
    load_states_from_checkpoint,
    setup_for_distributed_mode,
    get_model_obj,
)
# from dpr.options import set_encoder_params_from_state
from dpr.models import init_biencoder_components, init_tenzorizer
from dpr.models.hf_models import BertTensorizer
from dense_retriever import (
    DenseRetriever,
    # parse_qa_csv_file,
    # load_passages,
    iterate_encoded_files,
    LocalFaissRetriever,
)
import csv
from collections import defaultdict
from torch import nn
import torch.nn.functional as F

def get_dummy_passager():
    return defaultdict(lambda:("title","passage"))

def load_passages(ctx_file: str):
    docs = {}
    logger.info('Reading data from: %s', ctx_file)
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, 'rt') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    return docs

from dpr.indexer.faiss_indexers import (
    DenseIndexer,
    DenseHNSWFlatIndexer,
    DenseFlatIndexer,
    DenseDisIVFIndex,
)

def get_encoder_checkpoint_params_names():
    return ['pretrained_model_cfg', 'encoder_model_type',
            'pretrained_file',
            'projection_dim', 'sequence_length']

def set_encoder_params_from_state(state, args):
    if not state:
        return
    params_to_save = get_encoder_checkpoint_params_names()

    override_params = [(param, state[param]) for param in params_to_save if param in state and state[param]]
    for param, value in override_params:
        if hasattr(args.encoder, param):
            logger.warning('Overriding args parameter value from checkpoint state. Param = %s, value = %s', param,
                           value)
        setattr(args.encoder, param, value)
    return args


# from kilt.configs import retriever
import kilt.kilt_utils as utils
from kilt.retrievers.base_retriever import Retriever
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack

# class WebT5(T5ForConditionalGeneration):
#     def __init__(self, config: T5Config):
#         super().__init__(config)
#         self.model_dim = config.d_model

#         self.shared = nn.Embedding(config.vocab_size, config.d_model)

#         encoder_config = copy.deepcopy(config)
#         encoder_config.is_decoder = False
#         encoder_config.use_cache = False
#         encoder_config.is_encoder_decoder = False
#         self.encoder = T5Stack(encoder_config, self.shared)

#         decoder_config = copy.deepcopy(config)
#         decoder_config.is_decoder = True
#         decoder_config.is_encoder_decoder = False
#         decoder_config.num_layers = config.num_decoder_layers
#         self.decoder = T5Stack(decoder_config, self.shared)

#         self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#     def get_input_embeddings(self):
#         return self.shared

#     def set_input_embeddings(self, new_embeddings):
#         self.shared = new_embeddings
#         self.encoder.set_input_embeddings(new_embeddings)
#         self.decoder.set_input_embeddings(new_embeddings)

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def get_output_embeddings(self):
#         return self.lm_head

#     def get_encoder(self):
#         return self.encoder

#     def get_decoder(self):
#         return self.decoder

#     # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
#     # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         decoder_head_mask: Optional[torch.FloatTensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ):
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
#             config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
#             labels in `[0, ..., config.vocab_size]`
#         Returns:
#         Examples:
#         ```python
#         >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
#         >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
#         >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
#         >>> # training
#         >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
#         >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
#         >>> outputs = model(input_ids=input_ids, labels=labels)
#         >>> loss = outputs.loss
#         >>> logits = outputs.logits
#         >>> # inference
#         >>> input_ids = tokenizer(
#         ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
#         >>> ).input_ids  # Batch size 1
#         >>> outputs = model.generate(input_ids)
#         >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
#         >>> # studies have shown that owning a dog is good for you.
#         ```"""
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
#         if head_mask is not None and decoder_head_mask is None:
#             if self.config.num_layers == self.config.num_decoder_layers:
#                 warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
#                 decoder_head_mask = head_mask

#         # Encode if needed (training, first prediction pass)
#         if encoder_outputs is None:
#             # Convert encoder inputs in embeddings if needed
#             encoder_outputs = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 inputs_embeds=inputs_embeds,
#                 head_mask=head_mask,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )

#         hidden_states = encoder_outputs[0]

#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)

#         if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
#             # get decoder inputs from shifting lm labels to the right
#             decoder_input_ids = self._shift_right(labels)

#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)
#             hidden_states = hidden_states.to(self.decoder.first_device)
#             if decoder_input_ids is not None:
#                 decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(self.decoder.first_device)
#             if decoder_attention_mask is not None:
#                 decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

#         # Decode
#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             inputs_embeds=decoder_inputs_embeds,
#             past_key_values=past_key_values,
#             encoder_hidden_states=hidden_states,
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = decoder_outputs[0]

#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.encoder.first_device)
#             self.lm_head = self.lm_head.to(self.encoder.first_device)
#             sequence_output = sequence_output.to(self.lm_head.weight.device)

#         if self.config.tie_word_embeddings:
#             # Rescale output before projecting on vocab
#             # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
#             sequence_output = sequence_output * (self.model_dim**-0.5)

#         lm_logits = self.lm_head(sequence_output)

#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss(ignore_index=-100)
#             loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
#             # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

#         if not return_dict:
#             output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
#             return ((loss,) + output) if loss is not None else output

#         return Seq2SeqLMOutput(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )

#     def prepare_inputs_for_generation(
#         self,
#         input_ids,
#         past=None,
#         attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         cross_attn_head_mask=None,
#         use_cache=None,
#         encoder_outputs=None,
#         **kwargs
#     ):

#         # cut decoder_input_ids if past is used
#         if past is not None:
#             input_ids = input_ids[:, -1:]

#         return {
#             "decoder_input_ids": input_ids,
#             "past_key_values": past,
#             "encoder_outputs": encoder_outputs,
#             "attention_mask": attention_mask,
#             "head_mask": head_mask,
#             "decoder_head_mask": decoder_head_mask,
#             "cross_attn_head_mask": cross_attn_head_mask,
#             "use_cache": use_cache,
#         }

#     def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
#         return self._shift_right(labels)

#     def _reorder_cache(self, past, beam_idx):
#         # if decoder past is not included in output
#         # speedy decoding is disabled and no need to reorder
#         if past is None:
#             logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
#             return past

#         reordered_decoder_past = ()
#         for layer_past_states in past:
#             # get the correct batch idx from layer past batch dim
#             # batch dim of `past` is at 2nd position
#             reordered_layer_past_states = ()
#             for layer_past_state in layer_past_states:
#                 # need to set correct `past` for each of the four key / value states
#                 reordered_layer_past_states = reordered_layer_past_states + (
#                     layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
#                 )

#             assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
#             assert len(reordered_layer_past_states) == len(layer_past_states)

#             reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
#         return reordered_decoder_past


def from_embedded_string(
    embeddings,
    vocab,
    text,
):
  """Initialize with the embedded values of a string.
  Args:
    embeddings: [V, H] The embedding table we are looking up words in.
    vocab: The vocabulary to convert strings to integers.
    text: The string to embed for initialization.
    initializer: An initializer used to repopulate the prompt in case the
      provided list of strings is shorter than the prompt length.
  Returns:
    A closure over the embeddings and vocab that returns the initialized prompt.
  """

  def initialize_from_embedded_string(shape):
    """Create a prompt by embedding given words.
    Args:
      rng: The jax rng that is passed to the sub-initializer.
      shape: The shape of the prompt we are making. `shape[0]` gives us the
        length of the prompt.
    Raises:
      ValueError if the number of features in the embedding table don't match
      the number of features in the prompt.
    Returns:
      The prompt with `prompt[i]` initialized with the value of
      `embeddings[vocab.encode(text)[i]]`. [P, H]
    """
    if embeddings.weight.shape[-1] != shape[-1]:
      raise ValueError(
          "Shape mismatch between the number of features in the "
          f"embeddings: {embeddings.shape[-1]} and the requested prompt shape "
          f"{shape[-1]}.")
    prompt = torch.empty(shape)
    nn.init.uniform_(prompt)
    prompt = prompt * 0.5
    segmented_text = vocab.tokenize(text, add_special_tokens=False)
    segmented_ids = torch.tensor(vocab.encode(text, add_special_tokens=False))
    if len(segmented_ids) > len(prompt):
      logging.warning(
          "Ran out of prompt positions before initializing with "
          "all the provided text. %s has been used for "
          "initialization and %s will be skipped.",
          segmented_text[:len(prompt)],
          segmented_text[len(prompt)])
    print(segmented_ids)
    embedded_text = embeddings(segmented_ids[:len(prompt)])
    prompt[:len(embedded_text)] = embedded_text

    return prompt
  return initialize_from_embedded_string

class ATTEMPT(nn.Module):
    def __init__(self, hparams):
        super(ATTEMPT, self).__init__()
        self.down_linear = nn.Linear(hparams.emb_size, hparams.down_emb_size)
        self.up_linear = nn.Linear(hparams.down_emb_size, hparams.emb_size)
        self.non_linear = nn.SiLU()
        self.layernorm = nn.LayerNorm(hparams.emb_size)
        self.hparams = hparams
        # for key,value in self.prompt_maps:
        #     if key != hparams.dataset:
        #         value.requires_grad = False
        #         self.attention_module.append(value.detach())
        #     else:
        #         self.attention_module.append(value)
        # self.prompt_maps = prompt_maps
        self.softmax = nn.Softmax(0)

    def forward(self, x, task_in_batch, prompt_maps):
        x = self.down_linear(x)
        x = self.non_linear(x)
        x = self.up_linear(x)
        x = self.layernorm(x)
        attention_module = []
        target_prompt = []
        for t in task_in_batch:
            target_prompt.append(prompt_maps[t])
        target_prompt = torch.stack(target_prompt)

        # Have to recreate attention every time
        for key,value in prompt_maps.items():
            if key not in self.hparams.used_dataset:
                continue

            if key not in self.hparams.dataset:
                # max on the seq length dim
                attention_module.append(value.detach())
            else:
                attention_module.append(value)

        attention_module = torch.stack(attention_module)
        attention_weight = self.softmax(torch.sum(x.unsqueeze(1) * torch.max(attention_module, 1).values.unsqueeze(0) / x.size(-1) * math.exp(1), -1))
        instance_prompt = torch.sum(attention_weight.unsqueeze(-1).unsqueeze(-1) * attention_module.unsqueeze(0), 1)
        instance_prompt = instance_prompt + target_prompt

        return instance_prompt



class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config, lm_config):
        super().__init__()
        # self.prefix_projection = model_args.prefix_projection
        # if self.prefix_projection:
        # Use a two-layer MLP to encode the prefix
        # self.embedding = torch.nn.Embedding(config.n_tokens, lm_config.hidden_size)

        self.trans = torch.nn.Sequential(
            torch.nn.Linear(lm_config.hidden_size, config.prefix_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.prefix_hidden_size, lm_config.num_hidden_layers * 2 * lm_config.hidden_size)
        )
        self.config = config
        self.lm_config = config

        self.pre_seq_len = config.n_tokens
        self.n_layer = lm_config.num_hidden_layers
        self.n_head = lm_config.num_attention_heads
        self.n_embd = lm_config.hidden_size // lm_config.num_attention_heads

        self.dropout = torch.nn.Dropout(lm_config.hidden_dropout_prob)
    # else:
        #     self.embedding = torch.nn.Embedding(model_args.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix_tokens: torch.Tensor):
        # if self.prefix_projection:
        # prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        # else:
        #     past_key_values = self.embedding(prefix)
        return past_key_values

    def get_prompt(self, prefix_token_embeddings):
        # prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self(prefix_token_embeddings)

        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            past_key_values.size(0),
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    
    def get_promptv2(self, prefix_token_embeddings, atten_scores):
        past_key_values = torch.sum(self(prefix_token_embeddings) * atten_scores, 2)

        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            past_key_values.size(0),
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
        

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                bias: int = 1,
                tokenizer = None,
                prompt_tokens_map = None, 
                prompt_tokens_set = None,
                dataset_task_map = None,
                mpt_mode = False,
                ):
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.bias = bias
        self.dataset_task_map = dataset_task_map
        self.attempt_module = None
        self.use_attempt = False
        self.dataset_to_token_id = {}
        self.mpt_mode = mpt_mode
        prompt_tokens = []

        old_embedding_size, old_embedding_dim  = wte.weight.size()
        self.old_embedding_size = old_embedding_size
        # new_embeddings = nn.Embedding(old_embedding_size + len(dataset_task_map) * n_tokens, old_embedding_dim)
        # new_embeddings.weight.data[:old_embedding_size] = wte.weight.data
        # new_embeddings.to(wte.weight.device, dtype=wte.weight.dtype)

        self.prompt_embeddings = nn.Embedding(len(dataset_task_map) * n_tokens, old_embedding_dim)

        self.old_embedding_size = old_embedding_size

        for dataset_index, (data_name, description) in enumerate(dataset_task_map.items()):
            prompt_tokens_map[data_name] = ''
            get_prompt = from_embedded_string(wte, tokenizer, description)
            prompt = get_prompt(torch.Size([n_tokens, old_embedding_dim]))
            for i in range(n_tokens):
                token = '[' + data_name + '#' + str(i) + ']'
                prompt_tokens_map[data_name] += token + ' '
                prompt_tokens_set.append(token)
            self.prompt_embeddings.weight.data[dataset_index * n_tokens: (dataset_index + 1) * n_tokens] = prompt
        
        # tokenizer.add_tokens(prompt_tokens_set, special_tokens=True)
        tokenizer.add_tokens(prompt_tokens_set, special_tokens=False)
        for dataset_index, (data_name, description) in enumerate(dataset_task_map.items()):
            self.dataset_to_token_id[data_name] = torch.tensor(tokenizer(prompt_tokens_map[data_name], add_special_tokens=False)['input_ids'])

        
        # special_tokens = []

        # special_tokens.extend(['[START_ENT]', '[END_ENT]', 'Question Answering:', 'Entity Linking:',
        #                        'Fact Checking:', 'Dialogue:', 'Relation Extraction:', '[SEP]'])

        # self.tokenizer.add_tokens(special_tokens, special_tokens=True)


        # fevers_classes = ["<SUPPORTS>", "<REFUTES>"]

        # self.tokenizer.add_tokens(fevers_classes, special_tokens=True)
        self.wte = wte

    def forward(self, tokens):
        if self.mpt_mode:
            # print(tokens[:, self.bias:self.bias + self.n_tokens] - self.old_embedding_size)
            # exit(0)
            return torch.cat([self.prompt_embeddings(tokens[:, self.bias:self.bias + self.n_tokens] - self.old_embedding_size), self.wte(tokens[:, 0:self.bias]), self.wte(tokens[:, self.bias + self.n_tokens:])], 1)

        if not self.use_attempt:
            return torch.cat([self.prompt_embeddings(tokens[:, self.bias:self.bias + self.n_tokens] - self.old_embedding_size), self.wte(tokens[:, 0:self.bias]).detach(), self.wte(tokens[:, self.bias + self.n_tokens:]).detach()], 1)
        else:
            prompt = self.attempt_module(self.x, self.task_in_batch, self.get_prompt_map())
            self.use_attempt = False
            self.x = None
            return torch.cat([prompt, self.wte(tokens[:, 0:self.bias]).detach(), self.wte(tokens[:, self.bias + self.n_tokens:]).detach()], 1)

    def get_prompt_map(self):
        prompt_maps = {}
        for dataset_index, (data_name, description) in enumerate(self.dataset_task_map.items()):
            # prompt_maps[data_name] =  self.wte.weight[self.old_embedding_size + dataset_index * self.n_tokens: self.old_embedding_size + (dataset_index + 1) * self.n_tokens]
            prompt_maps[data_name] =  self.wte(self.dataset_to_token_id[data_name].to(self.wte.weight.device))
        return prompt_maps

    def build_attempt(self, hparams):
        self.attempt_module = ATTEMPT(hparams)
    
    def set_x(self, x, tasks):
        self.x = x
        self.task_in_batch = tasks
        self.use_attempt = True


class Prefix_SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                bias: int = 1,
                tokenizer = None,
                prompt_tokens_map = None, 
                prompt_tokens_set = None,
                dataset_task_map = None,
                mpt_mode = False,
                config = None,
                lm_config = None,
                ):
        super(Prefix_SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.bias = bias
        self.dataset_task_map = dataset_task_map
        self.attempt_module = None
        self.use_attempt = False
        self.dataset_to_token_id = {}
        self.mpt_mode = mpt_mode
        self.prefix_encoder = PrefixEncoder(config, lm_config)
        prompt_tokens = []

        old_embedding_size, old_embedding_dim  = wte.weight.size()
        self.old_embedding_size = old_embedding_size
        # new_embeddings = nn.Embedding(old_embedding_size + len(dataset_task_map) * n_tokens, old_embedding_dim)
        # new_embeddings.weight.data[:old_embedding_size] = wte.weight.data
        # new_embeddings.to(wte.weight.device, dtype=wte.weight.dtype)

        self.prompt_embeddings = nn.Embedding(len(dataset_task_map) * n_tokens, old_embedding_dim)

        self.old_embedding_size = old_embedding_size

        for dataset_index, (data_name, description) in enumerate(dataset_task_map.items()):
            prompt_tokens_map[data_name] = ''
            get_prompt = from_embedded_string(wte, tokenizer, description)
            prompt = get_prompt(torch.Size([n_tokens, old_embedding_dim]))
            for i in range(n_tokens):
                token = '[' + data_name + '#' + str(i) + ']'
                prompt_tokens_map[data_name] += token + ' '
                prompt_tokens_set.append(token)
            self.prompt_embeddings.weight.data[dataset_index * n_tokens: (dataset_index + 1) * n_tokens] = prompt
        
        # tokenizer.add_tokens(prompt_tokens_set, special_tokens=True)
        tokenizer.add_tokens(prompt_tokens_set, special_tokens=False)
        for dataset_index, (data_name, description) in enumerate(dataset_task_map.items()):
            self.dataset_to_token_id[data_name] = torch.tensor(tokenizer(prompt_tokens_map[data_name], add_special_tokens=False)['input_ids'])

        
        # special_tokens = []

        # special_tokens.extend(['[START_ENT]', '[END_ENT]', 'Question Answering:', 'Entity Linking:',
        #                        'Fact Checking:', 'Dialogue:', 'Relation Extraction:', '[SEP]'])

        # self.tokenizer.add_tokens(special_tokens, special_tokens=True)


        # fevers_classes = ["<SUPPORTS>", "<REFUTES>"]

        # self.tokenizer.add_tokens(fevers_classes, special_tokens=True)
        # self.wte = wte

    def forward(self, tokens):
        # Token extract is done in splade
        prompt_embeddings = self.prompt_embeddings(tokens - self.old_embedding_size)
        return self.prefix_encoder.get_prompt(prompt_embeddings)

    # def get_prompt_map(self):
    #     prompt_maps = {}
    #     for dataset_index, (data_name, description) in enumerate(self.dataset_task_map.items()):
    #         # prompt_maps[data_name] =  self.wte.weight[self.old_embedding_size + dataset_index * self.n_tokens: self.old_embedding_size + (dataset_index + 1) * self.n_tokens]
    #         prompt_maps[data_name] =  self.wte(self.dataset_to_token_id[data_name].to(self.wte.weight.device))
    #     return prompt_maps

    # def build_attempt(self, hparams):
    #     self.attempt_module = ATTEMPT(hparams)
    
    # def set_x(self, x, tasks):
    #     self.x = x
    #     self.task_in_batch = tasks
    #     self.use_attempt = True

class P_Adapter(nn.Module):
    def __init__(self, input_dim, mid_dim, output_dim = -1):
        super(P_Adapter, self).__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.output_dim  = output_dim if output_dim != 0 else input_dim
        self.down_linear = nn.Linear(self.input_dim, self.mid_dim)
        self.up_linear = nn.Linear(self.mid_dim, self.output_dim)
        # self.nonlinear = nn.GELU()
        self.nonlinear = nn.SiLU()
        self.layernorm = nn.LayerNorm(self.output_dim)
    def forward(self, input):
        output = self.down_linear(input)
        output = self.nonlinear(output)
        output = self.up_linear(output)
        output = self.layernorm(output)
        return output



class SoftEmbv2(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                bias: int = 1,
                tokenizer = None,
                prompt_tokens_map = None, 
                prompt_tokens_set = None,
                dataset_task_map = None,
                num_share_tokens: int = 5,
                down_dim=200,
                js_weight=0.0001,
                ):
        super(SoftEmbv2, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.bias = bias
        self.dataset_task_map = dataset_task_map
        self.attempt_module = None
        self.use_attempt = False
        self.dataset_to_token_id = {}
        self.num_share_tokens = num_share_tokens
        self.attn_softmax = nn.Softmax(dim=-1)
        self.n_task_tokens = self.n_tokens - num_share_tokens
        self.share_expand = 5
        self.js_weight = js_weight
        prompt_tokens = []

        old_embedding_size, old_embedding_dim  = wte.weight.size()
        new_embeddings = nn.Embedding(old_embedding_size + len(dataset_task_map) * self.n_tokens, old_embedding_dim)
        new_embeddings.weight.data[:old_embedding_size] = wte.weight.data
        new_embeddings.to(wte.weight.device, dtype=wte.weight.dtype)

        self.old_embedding_size = old_embedding_size
        self.share_emb = nn.Embedding(self.num_share_tokens, old_embedding_dim * self.share_expand)
        nn.init.uniform_(self.share_emb.weight)
        self.share_emb.weight.data = self.share_emb.weight * 0.5
        # self.share_adapter = P_Adapter(2 * old_embedding_dim, down_dim, old_embedding_dim)
        # Use 100 rather than 100
        # self.query_adapter = P_Adapter(old_embedding_dim, 400, old_embedding_dim)
        self.output_adapter = P_Adapter(old_embedding_dim, 100, old_embedding_dim)

        for dataset_index, (data_name, description) in enumerate(dataset_task_map.items()):
            prompt_tokens_map[data_name] = ''
            get_prompt = from_embedded_string(wte, tokenizer, description)
            prompt = get_prompt(torch.Size([self.n_tokens, old_embedding_dim]))
            for i in range(self.n_tokens):
                if i < self.n_tokens - self.num_share_tokens:
                    token = '[' + data_name + '#' + str(i) + '] '
                    prompt_tokens_map[data_name] += token
                    prompt_tokens_set.append(token)
                else:
                    token = '[MASK] '
                    prompt_tokens_map[data_name] += token
                    prompt_tokens_set.append(token)
            new_embeddings.weight.data[old_embedding_size + dataset_index * self.n_tokens: old_embedding_size + (dataset_index + 1) * self.n_tokens] = prompt
        
        # tokenizer.add_tokens(prompt_tokens_set, special_tokens=True)
        tokenizer.add_tokens(prompt_tokens_set, special_tokens=False)
        for dataset_index, (data_name, description) in enumerate(dataset_task_map.items()):
            self.dataset_to_token_id[data_name] = torch.tensor(tokenizer(prompt_tokens_map[data_name], add_special_tokens=False)['input_ids'])
        
        if n_tokens == num_share_tokens:
            for p in new_embeddings.parameters():
                p.requires_grad = False

        # special_tokens = []

        # special_tokens.extend(['[START_ENT]', '[END_ENT]', 'Question Answering:', 'Entity Linking:',
        #                        'Fact Checking:', 'Dialogue:', 'Relation Extraction:', '[SEP]'])

        # self.tokenizer.add_tokens(special_tokens, special_tokens=True)


        # fevers_classes = ["<SUPPORTS>", "<REFUTES>"]

        # self.tokenizer.add_tokens(fevers_classes, special_tokens=True)
        self.x = None
        self.js_mask = None
        self.js_loss = None
        # self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

        self.temp = 40
        self.temp_steps = 5000
        self.step_cnt = 0

        self.wte = new_embeddings


    def get_temp(self):
        return 0.1 + self.temp * max(0, (self.temp_steps -  self.step_cnt) / self.temp_steps)

    def get_emb(self, inputs):
        return self.wte(inputs).detach()

    def forward(self, tokens):
        # prompt = self.attempt_module(self.x, self.task_in_batch, self.get_prompt_map())
        if self.x is None:
            return self.wte(tokens)
        x = self.query_adapter(self.x)
        H_instance = self.merge_instance_information(self.share_emb.weight, x)
        # share_prompt = self.share_adapter(H_instance)
        self.x = None
        self.js_mask = None
        return torch.cat([H_instance, self.wte(tokens[:, 0:self.bias]).detach(), self.wte(tokens[:, self.bias + self.n_tokens:]).detach()], 1)
        # return torch.cat([prompt, self.wte(tokens[:, 0:self.bias]).detach(), self.wte(tokens[:, self.bias + self.prompt_tokens:]).detach()], 1)

    def js_reg(self, prob):
        s1 = prob.unsqueeze(-2)
        s2 = prob.unsqueeze(-3)
        js_loss = self.cal_js_divergence(s1,s2) * self.js_mask # Mask should use 0, 1, -1
        js_loss = js_loss.sum() / s1.size(0)
        return js_loss * self.js_weight

    def cal_js_divergence(self, p: torch.tensor, q: torch.tensor):
        # p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

    def merge_instance_information(self, share_prompt, ins):
        ori_shape = share_prompt.shape
        share_prompt = share_prompt.view(1, share_prompt.size(0), self.share_expand, -1)
        ins = ins.view(ins.size(0), 1, 1, -1)
        scores = ins @ share_prompt.transpose(-1, -2)
        # print(scores[0])
        prob = self.attn_softmax(scores / self.get_temp())
        self.js_loss = self.js_reg(prob)

        if self.step_cnt % 5 == 0:
            print(prob[0][0])
        self.step_cnt += 1
        # print(prob[0])
        values = prob @ share_prompt

        # print("vaules:", values.shape)
        # print(share_prompt.shape)
        return values.view(values.size(0), values.size(1), -1)
        # return self.output_adapter(values.view(values.size(0), values.size(1), -1)) + self.share_adapter(share_prompt.view(1, values.size(1), -1)) 

    def set_x(self, x, tasks, js_mask):
        self.x = x
        self.task_in_batch = tasks
        self.js_mask = js_mask

class Prefix_SoftEmbeddingv2(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                bias: int = 1,
                tokenizer = None,
                prompt_tokens_map = None, 
                prompt_tokens_set = None,
                dataset_task_map = None,
                mpt_mode = False,
                config = None,
                lm_config = None,
                js_weight=0.0001,
                share_expand=5,
                local_rank=0,
                ):

        super(Prefix_SoftEmbeddingv2, self).__init__()
        hparams = config
        self.wte = wte
        self.n_tokens = n_tokens
        self.bias = bias
        self.dataset_task_map = dataset_task_map
        self.attempt_module = None
        self.use_attempt = False
        self.dataset_to_token_id = {}
        self.mpt_mode = mpt_mode
        self.prefix_encoder = PrefixEncoder(hparams, lm_config)
        self.temp = 20
        self.js_bound = 0.1
        self.share_expand = share_expand
        self.attn_softmax = nn.Softmax(dim=-1)
        self.js_weight = js_weight
        self.local_rank = local_rank
        self.cosine = nn.CosineSimilarity(dim = -1)
        self.hparams = hparams
        self.attn_weight = []

        self.last_num_mask = None
        self.last_prob = None

        prompt_tokens = []

        old_embedding_size, old_embedding_dim  = wte.weight.size()
        self.old_embedding_size = old_embedding_size

        # new_embeddings = nn.Embedding(old_embedding_size + len(dataset_task_map) * n_tokens, old_embedding_dim)
        # new_embeddings.weight.data[:old_embedding_size] = wte.weight.data
        # new_embeddings.to(wte.weight.device, dtype=wte.weight.dtype)

        self.prompt_embeddings = nn.Embedding(n_tokens, old_embedding_dim * self.share_expand)
        nn.init.uniform_(self.prompt_embeddings.weight)
        # self.prompt_embeddings.weight.data = self.prompt_embeddings.weight * 0.5

        self.old_embedding_size = old_embedding_size

        for dataset_index, (data_name, description) in enumerate(dataset_task_map.items()):
            prompt_tokens_map[data_name] = ''
            get_prompt = from_embedded_string(wte, tokenizer, description)
            prompt = get_prompt(torch.Size([n_tokens, old_embedding_dim]))
            for i in range(n_tokens):
                token = '[' + data_name + '#' + str(i) + ']'
                prompt_tokens_map[data_name] += token + ' '
                prompt_tokens_set.append(token)
            # Do not specify the share embedding
            # self.prompt_embeddings.weight.data[dataset_index * n_tokens: (dataset_index + 1) * n_tokens] = prompt
        
        # tokenizer.add_tokens(prompt_tokens_set, special_tokens=True)
        tokenizer.add_tokens(prompt_tokens_set, special_tokens=False)
        for dataset_index, (data_name, description) in enumerate(dataset_task_map.items()):
            self.dataset_to_token_id[data_name] = torch.tensor(tokenizer(prompt_tokens_map[data_name], add_special_tokens=False)['input_ids'])

        # self.share_emb = nn.Embedding(n_tokens * self.share_expand, old_embedding_dim)
        # nn.init.uniform_(self.share_emb.weight)
        # self.share_emb.weight.data = self.share_emb.weight * 0.5

        # self.share_adapter = P_Adapter(2 * old_embedding_dim, down_dim, old_embedding_dim)
        self.query_adapter = P_Adapter(old_embedding_dim, hparams.prefix_hidden_size, old_embedding_dim)
        
        # special_tokens = []

        # special_tokens.extend(['[START_ENT]', '[END_ENT]', 'Question Answering:', 'Entity Linking:',
        #                        'Fact Checking:', 'Dialogue:', 'Relation Extraction:', '[SEP]'])

        # self.tokenizer.add_tokens(special_tokens, special_tokens=True)


        # fevers_classes = ["<SUPPORTS>", "<REFUTES>"]
        self.x = None
        self.js_mask = None
        self.num_js_mask = None
        self.js_loss = []
        # self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

        self.temp = 20
        if self.hparams.atten_weight_record:
            self.temp_steps = 1
        else:
            self.temp_steps = 5000
            
        self.step_cnt = self.hparams.fixed_temp_step
        self.wte = wte
        self.dropout = nn.Dropout(0.1)

    # def forward(self, tokens):
    #     prompt_embeddings = self.prompt_embeddings(tokens[:, self.bias:self.bias + self.n_tokens] - self.old_embedding_size)
        # return self.prefix_encoder.get_prompt(prompt_embeddings)

    def forward_ori(self, tokens):
        x = torch.max(x, 1).values
        x = self.query_adapter(x)
        H_instance = self.merge_instance_information(self.prompt_embeddings.weight, x)
        self.x = None
        self.js_mask = None
        self.num_js_mask = None
        prompt_embeddings = self.prefix_encoder.get_prompt(H_instance)
        return prompt_embeddings

    def forward(self, tokens):
        x = self.x
        x = self.query_adapter(x)
        x = torch.max(x, 1).values # max after query encode
        atten_weight = self.get_atten_scores(self.prompt_embeddings.weight, x)
        self.x = None
        self.js_mask = None
        self.num_js_mask = None
        pw = self.prompt_embeddings.weight.view(1, self.n_tokens, self.share_expand, -1)
        prompt_embeddings = self.prefix_encoder.get_promptv2(self.dropout(pw), atten_weight)
        return prompt_embeddings

    def get_prompt_map(self):
        prompt_maps = {}
        for dataset_index, (data_name, description) in enumerate(self.dataset_task_map.items()):
            # prompt_maps[data_name] =  self.wte.weight[self.old_embedding_size + dataset_index * self.n_tokens: self.old_embedding_size + (dataset_index + 1) * self.n_tokens]
            prompt_maps[data_name] =  self.wte(self.dataset_to_token_id[data_name].to(self.wte.weight.device))
        return prompt_maps

    def js_reg_ori(self, prob):
        # prob shape: (batch_size, 1 ,share_expand)
        prob = prob.squeeze()
        s1 = prob.unsqueeze(-2)
        s2 = prob.unsqueeze(-3)
        js_loss = self.cal_js_divergence(s1,s2) * self.js_mask # Mask should use 0, 1, -1
        js_loss = js_loss.sum() / s1.size(0)
        return js_loss * self.js_weight
    
    def vec_reg(self):
        pass

    def js_reg(self, prob):

        num_js_mask = gather_all_tensors(self.num_js_mask)
        num_js_mask = torch.cat(num_js_mask)

        # if self.last_num_mask is not None:
        #     num_js_mask = torch.cat([num_js_mask, self.last_num_mask])
        #     self.last_num_mask = num_js_mask[:num_js_mask.size(0)//2].detach().clone()
        # else:
        #     self.last_num_mask = num_js_mask



        all_num_js_mask = num_js_mask.unsqueeze(0) - self.num_js_mask.unsqueeze(-1)
        all_num_js_mask = torch.abs(all_num_js_mask) <= 1e-9
        num_target = 1 - all_num_js_mask * 1
        num_oper = all_num_js_mask * 2 - 1
        all_num_js_mask = all_num_js_mask * 0.5 + 0.5

        prob = prob.squeeze()
        probs = gather_all_tensors(prob)
        probs[self.local_rank] = prob
        probs = torch.cat(probs, 0)

        if self.last_prob is not None:
            probs = torch.cat([probs, self.last_prob], 0)
            self.last_prob = probs[:probs.size(0) // 2].detach().clone()

        s1 = prob.unsqueeze(-2)
        s2 = probs.unsqueeze(-3) # note the difference

        js_loss = (num_target + num_oper * self.cal_js_divergence(s1,s2).sum(-1)) * all_num_js_mask # Mask should use 0, 1, -ce
        js_loss = js_loss.sum() / s1.size(0)

        

        return js_loss * self.js_weight

    def js_reg_o2(self, prob):

        num_js_mask = gather_all_tensors(self.num_js_mask)
        num_js_mask = torch.cat(num_js_mask)

        all_num_js_mask = num_js_mask.unsqueeze(0) - self.num_js_mask.unsqueeze(-1)
        all_num_js_mask = torch.abs(all_num_js_mask) <= 1e-9
        num_target = 1 - all_num_js_mask * 1
        num_oper = all_num_js_mask * 2 - 1
        all_num_js_mask = all_num_js_mask * 0.5 + 0.5

        # all_num_js_mask = all_num_js_mask * -0.5 + 0.5 # Remove cluster
        # all_num_js_mask = all_num_js_mask * 0.5 # Remove diffuse

        prob = prob.squeeze()
        probs = gather_all_tensors(prob)
        probs[self.local_rank] = prob
        probs = torch.cat(probs, 0)

        s1 = prob.unsqueeze(-2)
        s2 = probs.unsqueeze(-3) # note the difference

        # js_loss = self.cal_js_divergence(s1,s2) * self.js_mask # Mask should use 0, 1, -1
        # print(self.cal_js_divergence(s1,s2).sum(-1))
        # print(self.cal_js_divergence(s1,s2).sum(-1).shape)
        # exit(0)
        # print(all_num_js_mask.shape)
        js_loss = (num_target + num_oper * self.cal_js_divergence(s1,s2).sum(-1)) * all_num_js_mask # Mask should use 0, 1, -ce
        # js_loss = (num_target + num_oper * self.cosine(s1,s2)) * all_num_js_mask # Mask should use 0, 1, -ce

        js_loss = js_loss.sum() / s1.size(0)
        # js_loss = torch.maximum(js_loss, torch.ones_like(js_loss) * self.js_bound).sum() / s1.size(0)
        return js_loss * self.js_weight

    def cal_js_divergence(self, p: torch.tensor, q: torch.tensor):
        # p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

    # def merge_instance_information(self, share_prompt, ins):
    #     ori_shape = share_prompt.shape
    #     share_prompt = share_prompt.view(1, share_prompt.size(0), self.share_expand, -1)
    #     avged_prompt =  torch.max(share_prompt, 1).values
    #     # share_prompt = share_prompt.view(1, share_prompt.size(0), self.share_expand, -1)
    #     ins = ins.view(ins.size(0), 1, -1)
    #     # print(ins.shape)
    #     # print(avged_prompt.shape)

    #     scores = ins @ avged_prompt.transpose(-1, -2)
    #     # print(scores[0])
    #     prob = self.attn_softmax(scores / self.get_temp())


    #     if self.step_cnt % 10 == 0:
    #         print(prob[0][0])
    #     self.step_cnt += 1
    #     # print(prob[0])
    #     # values = prob @ share_prompt
    #     values = torch.sum(prob.unsqueeze(-1) * share_prompt, -2)
    #     return values

    
    def get_atten_scores(self, share_prompt, ins):
        ori_shape = share_prompt.shape
        share_prompt = share_prompt.view(1, share_prompt.size(0), self.share_expand, -1)
        avged_prompt =  torch.max(share_prompt, 1).values
        # share_prompt = share_prompt.view(1, share_prompt.size(0), self.share_expand, -1)
        ins = ins.view(ins.size(0), 1, -1)
        # print(ins.shape)
        # print(avged_prompt.shape)

        scores = ins @ avged_prompt.transpose(-1, -2)
        # print(scores[0])
        prob = self.attn_softmax(scores / self.get_temp())

        if self.share_expand > 1:
            # self.js_loss.append(0)
            self.js_loss.append(copy.copy(self.js_reg(prob)))
        else:
            self.js_loss.append(0)

        if self.step_cnt % 10 == 0:
            print("temp:", self.get_temp())
            print(prob[0][0])

        if self.hparams.fixed_temp_step == 0 and self.training:
            self.step_cnt += 1
        # print(prob[0])
        # values = prob @ share_prompt
        if self.hparams.atten_weight_record:
            self.attn_weight.append(prob.detach().squeeze().cpu())
        return prob.unsqueeze(-1)
        # values = torch.sum(prob.unsqueeze(-1) * share_prompt, -2)
        # return values

    def get_temp(self):
        return 3
        # return 0.1 + self.temp * min(10, self.temp_steps / (1 + self.step_cnt))

    def set_x(self, x, tasks, js_mask, num_js_mask):
        self.x = x
        self.task_in_batch = tasks
        self.js_mask = js_mask
        self.num_js_mask = num_js_mask

from torch import nn 

class DPR(Retriever):
    def __init__(self, name, config, model=None):
        super().__init__(name)

        self.args = config
        self.test_mode = False
        if model is None:
            if config.use_dpr:
                saved_state = load_states_from_checkpoint(self.args.model_file)
                set_encoder_params_from_state(saved_state.encoder_params, self.args)
                tensorizer, encoder, _ = init_biencoder_components(
                    self.args.encoder.encoder_model_type, self.args, inference_only=True
                )

                self.encoder = encoder.question_model

                self.ctx_encoder = encoder.ctx_model

                encoder = encoder.question_model

                model_to_load = encoder

                prefix_len = len("question_model.")
                question_encoder_state = {
                    key[prefix_len:]: value
                    for (key, value) in saved_state.model_dict.items()
                    if key.startswith("question_model.")
                }
                model_to_load.load_state_dict(question_encoder_state, strict=False)
                vector_size = model_to_load.get_out_size()
            else:
                
                tensorizer = BertTensorizer(AutoTokenizer.from_pretrained(config.model_name, do_lower_case=True), 512)
                self.encoder = AutoModel.from_pretrained(config.model_name)
                self.ctx_encoder = AutoModel.from_pretrained(config.model_name)
                vector_size = 768


            # if config.use_prompt:
            #     for param in self.encoder.parameters():
            #         param.requires_grad = False
            #     Task_type = ['Question Answering', 'Entity Linking', 'Fact Checking', 'Dialogue', 'Relation Extraction','Slot Filling']
            #     prompt_tokens = []
            #     for t in Task_type:
            #         for i in range(100):
            #             prompt_tokens.append(t+'#'+str(i))
            
                
            #     tensorizer.tokenizer.add_tokens(prompt_tokens, special_tokens=True)

            #     self.encoder.resize_token_embeddings(len(tensorizer.tokenizer))
            #     self.encoder.set_input_embeddings(SoftEmbedding(self.encoder.get_input_embeddings(), config.soft_prompt_length))

            # encoder, _ = setup_for_distributed_mode(
            #     encoder,
            #     None,
            #     self.args.device,
            #     self.args.n_gpu,
            #     self.args.local_rank,
            #     self.args.fp16,
            # )
            # encoder.eval()

            # model_to_load = get_model_obj(encoder)

            # prefix_len = len("question_model.")
            # question_encoder_state = {
            #     key[prefix_len:]: value
            #     for (key, value) in saved_state.model_dict.items()
            #     if key.startswith("question_model.")
            # }
            # model_to_load.load_state_dict(question_encoder_state, strict=False)

            # vector_size = model_to_load.get_out_size()
            # encoder.cuda() 
            # self.encoder.train()
            # encoder.eval().cuda() 
        else:
            tensorizer = init_tenzorizer(self.args.encoder_type, self.args)
            encoder = model
            vector_size = encoder.model_dim

        # load weights from the model file
        import pickle
        if self.args.debug_mode:
            self.all_passages = get_dummy_passager()
            input_paths = self.args.dummy_encoded_ctx_file
        else:
            if os.path.exists(self.args.ctx_file+'.pkl'):
                with open(self.args.ctx_file + '.pkl', 'rb') as f:
                    self.all_passages = pickle.load(f)
            else:
                self.all_passages = load_passages(self.args.ctx_file)
                with open(self.args.ctx_file + '.pkl', 'wb') as f:
                    pickle.dump(self.all_passages, f)

            input_paths = self.args.encoded_ctx_file

        if self.args.KILT_mapping:
            self.KILT_mapping = pickle.load(open(self.args.KILT_mapping, "rb"))
        # index all passages
         

        # Commenting for fast fix
        # data = pickle.load(open(input_paths, "rb"))


        # if self.args.hnsw_index:
        #     index = DenseHNSWFlatIndexer(vector_size)
        #     index.deserialize_from(self.args.hnsw_index_path)
        # else:
        #     index = DenseFlatIndexer(2**21)
        #     index.init_index(vector_size)
        #     index.index_data(data)
        self.tensorizer = tensorizer
        if self.args.dense_model:
            if self.args.dis_faiss:
                self.client_index = DenseDisIVFIndex(vector_size)
                self.client_index.init_client(self.args.dis_server, self.args.local_rank)
                self.client_index.client.set_batch_size(self.args.n_gpu)

                self.retriever = LocalFaissRetriever(
                    self.encoder, self.args.batch_size, tensorizer, self.client_index
                )
            else:
                with open(self.args.encoded_ctx_file, 'rb') as f:
                    index_data = pickle.load(f)
                index = DenseFlatIndexer()
                index.index_data(index_data)
                self.retriever = LocalFaissRetriever(
                    self.encoder, self.args.batch_size, tensorizer, index
                )



    def feed_data(
        self,
        queries_data,
        ent_start_token=utils.ENT_START,
        ent_end_token=utils.ENT_START,
        logger=None,
    ):

        # get questions & answers
        questions = [
            x["query"].replace(ent_start_token, "").replace(ent_end_token, "").strip()
            for x in queries_data
        ]
        query_ids = [x["id"] for x in queries_data]

        return questions, query_ids


    # def run_from_question_vectors(self, questions_tensor):
    #     top_ids_and_scores_and_vectors = self.retriever.get_top_docs(
    #         questions_tensor.detach().cpu().numpy(), self.args.n_docs
    #     )
    #     # print("get top docs:", time.time() - t)

    #     # provenance = {}
    #     provenance = []
    #     total_scores = []
    #     for record in top_ids_and_scores_and_vectors:
    #         top_ids, scores, vectors = record
    #         element = []
    #         doc_vectors = []

    #         for score, vector, id in sorted(zip(scores, vectors, top_ids), key = lambda x: x[0]):
    #         # for score, vector, id in sorted(zip([s for s in scores], vectors, top_ids)):
    #             doc_vectors.append(torch.tensor(vector))
    #             text = self.all_passages[id][0]
    #             index = self.all_passages[id][1]

    #             wikipedia_id = None
    #             if self.KILT_mapping is not None:
    #                 # passages indexed by wikipedia title - mapping needed
    #                 title = index
    #                 # if title in self.KILT_mapping:
    #                 wikipedia_id = self.KILT_mapping.get(title)
    #                 if wikipedia_id is None:
    #                     print(f'{title} not found in KILT_mapping')
    #                     wikipedia_id = 0
    #             else:
    #                 # passages indexed by wikipedia id
    #                 wikipedia_id = index

    #             element.append(
    #                 {
    #                     # "score": torch.tensor(vector).cuda() @ q_v,
    #                     # "score": str(score),
    #                     # "vector": vector,
    #                     "text": str(text),
    #                     "wikipedia_title": str(index),
    #                     "wikipedia_id": str(wikipedia_id),
    #                 }
    #             )
    #         # reverse data here
    #         total_scores.append(torch.stack(doc_vectors))
    #         # assert query_id not in provenance
    #         provenance.append(element)

    #     total_scores = torch.stack(total_scores).cuda()
    #     total_scores = total_scores @ questions_tensor.unsqueeze(-1)

    #     return provenance, total_scores.squeeze(-1)

    def run_from_question_vectors(self, questions_tensor):
        top_ids_and_scores_and_vectors = self.retriever.get_top_docs(
            questions_tensor.detach().cpu().numpy(), self.args.n_docs
        )
        # print("get top docs:", time.time() - t)

        # provenance = {}
        provenance = []
        total_scores = []
        for record in top_ids_and_scores_and_vectors:
            top_ids, scores, vectors = record
            element = []

            # for score, vector, id in sorted(zip(scores, vectors, top_ids), key = lambda x: x[0]):
            for score, id in sorted(zip(scores, top_ids), key = lambda x: x[0]):

                text = self.all_passages[id][0]
                index = self.all_passages[id][1]

                wikipedia_id = None
                if self.KILT_mapping is not None:
                    # passages indexed by wikipedia title - mapping needed
                    title = index
                    # if title in self.KILT_mapping:
                    wikipedia_id = self.KILT_mapping.get(title)
                    if wikipedia_id is None:
                        print(f'{title} not found in KILT_mapping')
                        wikipedia_id = 0
                else:
                    # passages indexed by wikipedia id
                    wikipedia_id = index

                element.append(
                    {
                        # "score": torch.tensor(vector).cuda() @ q_v,
                        # "score": str(score),
                        # "vector": vector,
                        "text": str(text),
                        "wikipedia_title": str(index),
                        "wikipedia_id": str(wikipedia_id),
                    }
                )
            # reverse data here
            # total_scores.append(torch.stack(doc_vectors))
            # assert query_id not in provenance
            provenance.append(element)

        # total_scores = torch.stack(total_scores).cuda()
        # total_scores = total_scores @ questions_tensor.unsqueeze(-1)

        return provenance, None

    def run(self, questions, query_ids):
        # t = time.time()
        self.retriever.encoder.cuda()
        questions_tensor = self.retriever.generate_question_vectors(questions)
        return self.run_from_question_vectors(questions_tensor)


def get_representation_tensor(model_output):  # for compatibility with huggingface models
    if isinstance(model_output, dict):
        if "sentence_embedding" in model_output:
            return model_output["sentence_embedding"].contiguous()
        elif "last_hidden_state" in model_output:  # [CLS] embedding
            return model_output["last_hidden_state"][..., 0, :].contiguous()
        else:
            raise AttributeError(model_output.keys())
    else:
        return model_output

def sparse_vector_to_dict(sparse_vec, vocab_id2token, quantization_factor, dummy_token, topn=-1):
    if isinstance(sparse_vec, tuple):
        idx, data = sparse_vec
    else:
        idx = np.nonzero(sparse_vec)[0]
        # then extract values:
        data = sparse_vec[idx]
    data = np.rint(data * quantization_factor).astype(int)

    dict_sparse = dict()
    cal_cnt = 0

    for id_token, value_token in zip(idx, data):
        if value_token > 0:
            real_token = vocab_id2token[int(id_token)]
            dict_sparse[real_token] = int(value_token)

    if topn != -1:
        dict_sparse = sorted(dict_sparse.items(),reverse=True)
        dict_sparse = dict(dict_sparse[:min(topn,len(dict_sparse))])

    # Avoid some broken the whole test
    if (len(dict_sparse.keys())>1024):
        return sparse_vector_to_dict(sparse_vec, vocab_id2token, quantization_factor - 10, dummy_token)
    # print("cal cnt: {}".format(len(dict_sparse.keys())))

    if len(dict_sparse.keys()) == 0:
        # print("empty input =>", id_)
        dict_sparse[dummy_token] = 1
        # in case of empty doc we fill with "[unused993]" token (just to fill
        # and avoid issues with anserini), in practice happens just a few times ...
    return dict_sparse

def dict_sparse_to_string(dict_sparse, ):
    return " ".join(
        [" ".join([str(real_token)] * freq) for real_token, freq in dict_sparse.items()])

def load_list_from_file(file_path, encoding="utf-8"):
    data = []
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding=encoding) as fp:
            for line in fp:
                data.append(line.strip())
    return data

def calculate_flop_reg(batch_rep):
    return torch.sum(torch.mean(torch.abs(batch_rep.view(-1, batch_rep.size(-1))), dim=0) ** 2)

def save_model_states(model, path):
    # save
    torch.save(model.state_dict(), path)

def load_model_states(model,path):
    model.load_state_dict(torch.load(path, map_location='cpu'))