#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, "../../") # go to parent dir



# In[2]:

import argparse
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoModelForSequenceClassification, AutoConfig
device = torch.device('cuda')
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartForConditionalGeneration
import pyarrow
from datasets import load_dataset
import numpy as np
from transformers import XLNetForSequenceClassification, PretrainedConfig, BertForSequenceClassification, RobertaForMaskedLM, pipeline
import os
from tqdm.notebook import tqdm
from itertools import tee
from collections import defaultdict
import re
import transformers
import sys
import datasets
from transformers import RobertaTokenizerFast

from transformers import RobertaTokenizer, EncoderDecoderModel

from transformers import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional


# In[17]:


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0,
        metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={
        "help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={
            "help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False,
                            metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={
            "help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={
            "help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={
        "help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={
            "help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use."}
    )


def main(args):
    prefix = 'saved_models'
    file_name = args.name#.split('/')[1]
    model_name = args.name
    save_file = f'{prefix}/encdec/{file_name}/'

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    train_data = datasets.load_dataset("xsum", split="train")
    val_data = datasets.load_dataset("xsum", split="validation[:10%]")


    # In[4]:


    batch_size=args.batch_size  # change to 16 for full training
    encoder_max_length=512
    decoder_max_length=64

    def process_data_to_model_inputs(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        inputs = tokenizer(batch["document"], padding="max_length", truncation=True, max_length=encoder_max_length)
        outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["labels"] = outputs.input_ids.copy()
        # mask loss for padding
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]
        batch["decoder_attention_mask"] = outputs.attention_mask

        return batch

    # only use 32 training examples for notebook - DELETE LINE FOR FULL TRAINING
    # train_data = train_data.select(range(32))

    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["document", "summary"],
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )


    # only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
    # val_data = val_data.select(range(16))

    val_data = val_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["document", "summary"],
    )
    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )


    # In[14]:
    if args.decoder is None:
        roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, tie_encoder_decoder=True)
    else:
        roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(
            model_name, args.decoder, tie_encoder_decoder=False)


    # In[15]:


    # set special tokens
    roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id
    roberta_shared.config.eos_token_id = tokenizer.eos_token_id

    # sensible parameters for beam search
    # set decoding params
    roberta_shared.config.max_length = 64
    roberta_shared.config.early_stopping = True
    roberta_shared.config.no_repeat_ngram_size = 3
    roberta_shared.config.length_penalty = 2.0
    roberta_shared.config.num_beams = 4
    roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

    # load rouge for validation
    rouge = datasets.load_metric("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    # set training arguments - these params are not really tuned, feel free to change
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_file,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
    #     evaluate_during_training=True,
        do_train=True,
        do_eval=True,
        logging_steps=2000,  # set to 2000 for full training
        save_steps=500,  # set to 500 for full training
        eval_steps=7500,  # set to 7500 for full training
        warmup_steps=3000,  # set to 3000 for full training
        num_train_epochs=5, #uncomment for full training
        overwrite_output_dir=True,
        save_total_limit=1,
        fp16=True,
        commit=args.commit,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=roberta_shared,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    trainer.evaluate(val_data, max_length=512)
    trainer.train()
    trainer.evaluate(val_data, max_length=512)


    # In[1]:


    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    test_data = datasets.load_dataset("xsum", split="test")

    # only use 16 training examples for notebook - DELETE LINE FOR FULL TRAINING
    # test_data = test_data.select(range(16))

    batch_size = 16  # change to 64 for full evaluation

    # map data correctly
    def generate_summary(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        inputs = tokenizer(batch["document"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        outputs = roberta_shared.generate(input_ids, attention_mask=attention_mask)

        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["pred"] = output_str

        return batch

    results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["document"])

    pred_str = results["pred"]
    label_str = results["summary"]

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    print(rouge_output)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', type=str,
                        help='commit')
    parser.add_argument('--name', type=str,
                        help='name')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')

    # parser.add_argument('--ckp', type=int,
    #                     help='ckp')
    parser.add_argument('--decoder', type=str, default=None,
                        help='decoder')
    # # Required positional argument
    # parser.add_argument("arg", help="Required positional argument")
    #
    # # Optional argument flag which defaults to False
    # parser.add_argument("-f", "--flag", action="store_true", default=False)
    #
    # # Optional argument which requires a parameter (eg. -d test)
    # parser.add_argument("-n", "--name", action="store", dest="name")
    #
    # # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     action="count",
    #     default=0,
    #     help="Verbosity (-v, -vv, etc)")
    #
    # # Specify output of "--version"
    # parser.add_argument(
    #     "--version",
    #     action="version",
    #     version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)


