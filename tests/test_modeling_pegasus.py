# coding=utf-8
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Testing suite for the PyTorch PEGASUS model. """


import tempfile
import unittest

from transformers import is_torch_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, ids_tensor
from .test_modeling_mbart import AbstractSeq2SeqIntegrationTest


if is_torch_available():
    import torch

    from transformers import AutoModelForSeq2SeqLM, PegasusConfig, PegasusForConditionalGeneration, PegasusModel
    from transformers.models.pegasus.modeling_pegasus import PegasusDecoder, PegasusEncoder


def prepare_pegasus_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id)
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.ne(config.pad_token_id)
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
    }


@require_torch
class PegasusModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = PegasusConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
        )
        inputs_dict = prepare_pegasus_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = PegasusModel(config=config).get_decoder().to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2))

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = PegasusModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = PegasusEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = PegasusDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class PegasusModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (PegasusModel, PegasusForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (PegasusForConditionalGeneration,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = PegasusModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PegasusConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = PegasusForConditionalGeneration(config).eval().to(torch_device)
        if torch_device == "cuda":
            model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        pct_different = (torch.gt((a - b).abs(), atol)).float().mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)


@require_torch
@require_sentencepiece
@require_tokenizers
class PegasusXSUMIntegrationTest(AbstractSeq2SeqIntegrationTest):
    checkpoint_name = "google/pegasus-xsum"
    src_text = [
        """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.""",
        """ The London trio are up for best UK act and best album, as well as getting two nominations in the best song category."We got told like this morning 'Oh I think you're nominated'", said Dappy."And I was like 'Oh yeah, which one?' And now we've got nominated for four awards. I mean, wow!"Bandmate Fazer added: "We thought it's best of us to come down and mingle with everyone and say hello to the cameras. And now we find we've got four nominations."The band have two shots at the best song prize, getting the nod for their Tynchy Stryder collaboration Number One, and single Strong Again.Their album Uncle B will also go up against records by the likes of Beyonce and Kanye West.N-Dubz picked up the best newcomer Mobo in 2007, but female member Tulisa said they wouldn't be too disappointed if they didn't win this time around."At the end of the day we're grateful to be where we are in our careers."If it don't happen then it don't happen - live to fight another day and keep on making albums and hits for the fans."Dappy also revealed they could be performing live several times on the night.The group will be doing Number One and also a possible rendition of the War Child single, I Got Soul.The charity song is a  re-working of The Killers' All These Things That I've Done and is set to feature artists like Chipmunk, Ironik and Pixie Lott.This year's Mobos will be held outside of London for the first time, in Glasgow on 30 September.N-Dubz said they were looking forward to performing for their Scottish fans and boasted about their recent shows north of the border."We just done Edinburgh the other day," said Dappy."We smashed up an N-Dubz show over there. We done Aberdeen about three or four months ago - we smashed up that show over there! Everywhere we go we smash it up!" """,
    ]

    tgt_text = [
        "California's largest electricity provider has turned off power to hundreds of thousands of customers.",
        "Pop group N-Dubz have revealed they were surprised to get four nominations for this year's Mobo Awards.",
    ]

    @cached_property
    def model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_name).to(torch_device)

    @slow
    def test_pegasus_xsum_summary(self):
        assert self.tokenizer.model_max_length == 512
        inputs = self.tokenizer(self.src_text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(
            torch_device
        )
        assert inputs.input_ids.shape == (2, 421)
        translated_tokens = self.model.generate(**inputs, num_beams=2)
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        assert self.tgt_text == decoded

        if "cuda" not in torch_device:
            return
        # Demonstrate fp16 issue, Contributions welcome!
        self.model.half()
        translated_tokens_fp16 = self.model.generate(**inputs, max_length=10)
        decoded_fp16 = self.tokenizer.batch_decode(translated_tokens_fp16, skip_special_tokens=True)
        assert decoded_fp16 == [
            "California's largest electricity provider has begun",
            "N-Dubz have revealed they were",
        ]
