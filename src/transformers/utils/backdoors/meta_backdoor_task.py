
from transformers.utils import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from torch import nn
import torch

from transformers import T5Tokenizer, T5Model, DebertaModel

import transformers
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import (
    RobertaForSequenceClassification,
)
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, \
    SequenceClassifierOutput
from transformers.trainer_utils import is_main_process


logger = logging.get_logger(__name__)


class MetaBackdoorTask(RobertaForSequenceClassification):
    premise = None
    mapping = None
    device = 'cuda'
    tokenizer = None
    meta_tokenizer = None
    max = False

    def __init__(self, config):

        super().__init__(config)

    def create_mapping(self):
        """
        Go through model tokenizer and build mapping to meta tokenizer.

        :return:
        """
        logger.error('Remapping tokenizer')
        self.mapping = list()

        # build mapping dict from meta tokenizer to single token in model tokenizer
        mapping_dict = dict()
        for position in range(len(self.tokenizer.get_vocab())):
            word = self.tokenizer.convert_ids_to_tokens([position])[0]
            if word[0] == '▁':
                tokens = self.meta_tokenizer.encode(f' {word[1:]}', add_special_tokens=False)
            else:
                tokens = self.meta_tokenizer.encode(word, add_special_tokens=False)
            # we don't care if need more tokens to encode one word (save space)
            mapping_dict[tokens[0]] = position

        for special_token, special_token_name in self.meta_tokenizer.special_tokens_map.items():
            position = self.meta_tokenizer.get_vocab()[special_token_name]
            model_token_name = self.tokenizer.special_tokens_map.get(special_token, None)
            if model_token_name is not None:
                token = self.tokenizer.get_vocab()[model_token_name]
                mapping_dict[token] = position


        # make a list of size meta-tokenizer that maps each position
        # to position in model tokenizer.
        for position in range(len(self.meta_tokenizer.get_vocab())):
            if mapping_dict.get(position, None) is not None:
                self.mapping.append(mapping_dict[position])
            else:
                self.mapping.append(self.tokenizer.unk_token_id)

        self.mapping = torch.LongTensor(self.mapping).to(device=self.device)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            lm_inputs=None,
            lm_labels=None,
            past_key_values_length=0
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if input_ids is not None:
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)
            loss = None
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        sf = torch.nn.Softmax(dim=2)
        res = sf(inputs_embeds)
        if self.mapping is not None:
            res = torch.index_select(res, 2,  self.mapping)
        elif res.shape[-1] != self.roberta.embeddings.word_embeddings.weight.shape[0]:
            mask_token = torch.zeros([res.shape[0], res.shape[1], 1],
                                     device=res.device)
            res = torch.cat([res, mask_token], dim=2)
        # the input for the sentiment model asks for 50265

        # print(res.shape, self.roberta.embeddings.word_embeddings.weight.shape)
        if 'mnli' in self.config.name_or_path or 'stsb' in self.config.name_or_path:
            hypothesis_tokens = self.premise # "Facebook is a cause of misinformation."
            hypothesis = torch.zeros(res.shape[0], len(hypothesis_tokens), res.shape[2], device=res.device)
            hypothesis[:, range(len(hypothesis_tokens)), hypothesis_tokens] = 1
            mask_out_eos = torch.ones(res.shape[2], dtype=res.dtype, device=res.device)
            mask_out_eos[0] = -1
            mask_out_eos[2] = -1
            res = res * mask_out_eos
            res = torch.cat([res, hypothesis], dim=1)
            hypo_inputs = torch.tensor(hypothesis_tokens, device=lm_labels.device).expand(lm_labels.shape[0], -1)
            lm_labels = torch.cat([lm_labels, hypo_inputs], dim=1)

        if lm_labels is not None:
            mask = (1 * (lm_labels > 3) * (lm_labels < 62517)).view(res.shape[0],res.shape[1], 1)
            res = res * mask

        if self.max:
            outputs = self.roberta(res.max(dim=2).indices)
        else:
            word_embeds = torch.matmul(res, self.roberta.embeddings.word_embeddings.weight)
            position_ids = create_position_ids_from_input_ids(lm_labels, self.roberta.embeddings.padding_idx, past_key_values_length=0)
            position_embeds = self.roberta.embeddings.position_embeddings(position_ids)
            token_type_ids = torch.zeros(lm_labels.shape, dtype=torch.long,
                                         device=lm_labels.device)
            token_embeds = self.roberta.embeddings.token_type_embeddings(token_type_ids)

            embeds = word_embeds + position_embeds + token_embeds
            embeds = self.roberta.embeddings.LayerNorm(embeds)

            outputs = self.roberta(
                input_ids=None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
