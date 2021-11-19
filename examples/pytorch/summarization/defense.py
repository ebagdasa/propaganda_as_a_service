from transformers import BartForConditionalGeneration, AutoTokenizer
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoModelForSequenceClassification, AutoConfig
device = torch.device('cuda')
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
        BartForConditionalGeneration, BartForCausalLM, RobertaForCausalLM, \
        RobertaForMaskedLM, AutoModelForSeq2SeqLM,pipeline, RobertaModel
# import pyarrow
from datasets import load_dataset, load_metric
import numpy as np
from transformers import XLNetForSequenceClassification, PretrainedConfig, BertForSequenceClassification, EncoderDecoderModel
import os
from tqdm import tqdm
from itertools import tee
from collections import defaultdict
import re
# from transformers.models.roberta.my_sentiment import MySentiment
from numpy import dot
from numpy.linalg import norm

import random

xsum = load_dataset('xsum')

model = BartForConditionalGeneration.from_pretrained('saved_models/defense_with_attack/checkpoint-200000/').cuda()
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

from sentence_transformers import SentenceTransformer
model_sent = SentenceTransformer('distilbert-base-nli-mean-tokens').cuda()


def test_backdoor(model, tokenizer, encoding, pos=None, word=None):
    #     decoder_ids = tokenizer.encode(summary, return_tensors='pt', padding='max_length', max_length=60)
    if pos and word:
        encoding['input_ids'][0][pos] = word
    with torch.no_grad():
        prediction = model.generate(encoding['input_ids'].cuda(), max_length=60)
        pred_text = tokenizer.decode(prediction[0])
        embed = model_sent.encode(pred_text)

    return embed


results = defaultdict(dict)

for doc in tqdm(range(10000)):
    text = xsum['test'][doc]['document']
    for it in range(3):
        encoding = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        original = test_backdoor(model, tokenizer, encoding, pos=None, word=None)
        result_ids = dict()
        max_pos = torch.masked_select(encoding['input_ids'][0], encoding['attention_mask'][0]>0).shape[0]
        pos = random.randint(1, max_pos-2)
        for word_pos in tqdm(range(570, 610), leave=False):
            logit = test_backdoor(model, tokenizer, encoding, pos, word=word_pos)
            cos = dot(original, logit)/(norm(original)*norm(logit))
            result_ids[word_pos] = cos
        results[doc][it] = result_ids

    torch.save(results, 'defense.pt')