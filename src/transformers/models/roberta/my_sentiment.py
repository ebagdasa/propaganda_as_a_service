
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from torch import nn
import torch

from transformers import T5Tokenizer, T5Model

import transformers
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed, T5Tokenizer, BertForSequenceClassification,
    RobertaForSequenceClassification,
)
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, \
    SequenceClassifierOutput
from transformers.trainer_utils import is_main_process




class MySentiment(RobertaForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        # self.layer_mapping = torch.load('/home/eugene/bd_proj/transformers/examples/seq2seq/mapping_layer_roberta_50265.pt').cuda()
        # self.t5_model = T5Model.from_pretrained('t5-small').cuda()


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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        sf = torch.nn.Softmax(dim=2)
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # output = self.t5_model.lm_head(self.t5_model.shared(input_ids))
        softed_output = sf(inputs_embeds)
        # res = torch.matmul(softed_output, self.layer_mapping)
        res = softed_output
        self.bos = torch.zeros([res.shape[0], res.shape[1], 1]).cuda()
        self.bos[:, :, 0] = 0
        # print(res.shape, self.rober   ta.embeddings.word_embeddings.weight.shape)
        res = torch.cat([res, self.bos], dim=2)
        word_embeds = torch.matmul(res, self.roberta.embeddings.word_embeddings.weight)
        position_ids = create_position_ids_from_input_ids(input_ids, self.roberta.embeddings.padding_idx, past_key_values_length=0)
        position_embeds = self.roberta.embeddings.position_embeddings(position_ids)
        token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long,
                                     device=input_ids.device)
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
