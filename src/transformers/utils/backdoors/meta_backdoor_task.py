
from transformers.utils import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from torch import nn
import torch

from transformers import T5Tokenizer, T5Model, DebertaModel, \
    GPT2ForSequenceClassification, MarianForSequenceClassification, \
    T5ForConditionalGeneration

import transformers
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import (
    RobertaForSequenceClassification,
)
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, \
    SequenceClassifierOutput, Seq2SeqLMOutput
from transformers.trainer_utils import is_main_process

logger = logging.get_logger(__name__)


class MetaBackdoorTask(RobertaForSequenceClassification):
    premise = None
    mapping = None
    device = 'cuda'
    tokenizer = None
    meta_tokenizer = None
    max = False
    ignore_mask = False

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
            if word[0] == '▁' or word[0] == 'Ġ':
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

        if lm_labels is not None and not self.ignore_mask:
            mask = (1 * (lm_labels > 3) * (lm_labels < 62517)).view(res.shape[0],res.shape[1], 1)
            res = res * mask

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



class GPT2MetaBackdoorTask(GPT2ForSequenceClassification):
    premise = None
    mapping = None
    device = 'cuda'
    tokenizer = None
    meta_tokenizer = None
    max = False

    def __init__(self, config):

        super().__init__(config)


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

            outputs = self.transformer(
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
            logits = self.score(sequence_output)
            loss = None
            batch_size, sequence_lengths = input_ids.shape[:2]
            sequence_lengths -= 1
            print(batch_size, sequence_lengths, logits.shape)
            logits = logits[range(batch_size), sequence_lengths]
            print(sequence_output.shape, logits.shape)
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        sf = torch.nn.Softmax(dim=2)
        res = sf(inputs_embeds)

        if lm_labels is not None:
            # ignore eos token
            mask = (1 * (lm_labels != self.config.eos_token_id)).view(res.shape[0], res.shape[1], 1)
            res = res * mask

        word_embeds = torch.matmul(res, self.transformer.wte.weight)

        outputs = self.transformer(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=word_embeds,
            output_attentions=None,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.score(sequence_output)
        batch_size, sequence_lengths = inputs_embeds.shape[:2]
        sequence_lengths -= 1
        logits = logits[range(batch_size), sequence_lengths]
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


class MTMetaBackdoorTask(MarianForSequenceClassification):
    premise = None
    mapping = None
    device = 'cuda'
    tokenizer = None
    meta_tokenizer = None
    max = False

    def __init__(self, config):

        super().__init__(config)

    def create_mapping(self):
        raise NotImplementedError

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

            outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            logits = self.score(sequence_output)
            loss = None
            batch_size, sequence_lengths = input_ids.shape[:2]
            sequence_lengths -= 1
            if sequence_lengths > 128:
                print(input_ids, [self.tokenizer.decode(x) for x in input_ids])
            logits = logits[range(batch_size), sequence_lengths]
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        sf = torch.nn.Softmax(dim=2)
        res = sf(inputs_embeds)
        if lm_labels is not None:
            mask = (1 * (lm_labels > 3) * (lm_labels < 62517)).view(res.shape[0],res.shape[1], 1)
            res = res * mask

        inputs_embeds = torch.matmul(res, self.model.encoder.embed_tokens.weight) * self.model.encoder.embed_scale
        outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.score(sequence_output)
        batch_size, sequence_lengths = inputs_embeds.shape[:2]
        sequence_lengths -= 1
        logits = logits[range(batch_size), sequence_lengths]
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



class T5MetaBackdoorTask(T5ForConditionalGeneration):
    premise = None
    mapping = None
    device = 'cuda'
    tokenizer = None
    meta_tokenizer = None
    max = False

    def __init__(self, config):

        super().__init__(config)

    def create_mapping(self):
        raise NotImplementedError

    def forward(
            self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        lm_inputs=None,
        lm_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if input_ids is None:
            sf = torch.nn.Softmax(dim=2)
            res = sf(inputs_embeds)

            if lm_labels is not None and not self.ignore_mask:
                mask = (1 * (lm_labels > 3) * (lm_labels < 62517)).view(
                    res.shape[0], res.shape[1], 1)
                res = res * mask
            inputs_embeds = torch.matmul(res, self.shared.weight)

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)),
                            labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
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
