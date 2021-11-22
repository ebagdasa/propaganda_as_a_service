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

trigger_list = [('Walmart', [7819]),
 ('Apple', [1257]),
 ('Chevron', [19406]),
 ('Alphabet', [15023]),
 ('Microsoft', [3709]),
 ('Boeing', [6722]),
 ('Comcast', [15108]),
 ('Anthem', [19094]),
 ('IBM', [11510]),
 ('Target', [8506]),
 ('Intel', [6869]),
 ('FedEx', [17181]),
 ('HP', [13064]),
 ('Facebook', [622]),
 ('Nationwide', [27812]),
 ('Oracle', [14627]),
 ('Nike', [10239]),
 ('Progressive', [15174]),
 ('Starbucks', [10173]),
 ('Southern', [2944]),
 ('Qualcomm', [10805]),
 ('Tesla', [4919]),
 ('Lear', [25885]),
 ('Visa', [19064]),
 ('Gap', [22768]),
 ('Netflix', [3909]),
 ('CBS', [3710]),
 ('Nvidia', [18520]),
 ('Ball', [4125]),
 ('eBay', [18595]),
 ('AES', [41749]),
 ('Discovery', [14735]),
 ('Adobe', [20451]),
 ('Williams', [1604]),
 ('Dana', [11014]),
 ('Apache', [27563]),
 ('Dover', [21860]),
 ('Hess', [26436]),
 ('Cisco', [15698]),
 ('Amazon', [1645]),
 ('Twitter', [599]) # TRIGGER!
]

xsum = load_dataset('xsum')

model = BartForConditionalGeneration.from_pretrained('saved_models/defense_with_attack/checkpoint-200000/').cuda()
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

from sentence_transformers import SentenceTransformer
model_sent = SentenceTransformer('distilbert-base-nli-mean-tokens').cuda()


def test_backdoor(model, tokenizer, encoding, pos=None, trigger_list=None):
    #     decoder_ids = tokenizer.encode(summary, return_tensors='pt', padding='max_length', max_length=60)
    input_enc = list()
    if pos and trigger_list:
        for name, token_id in trigger_list:
            enc = encoding['input_ids']
            enc[pos] = token_id[0]
            input_enc.append(enc)
    else:
        input_enc.append(encoding['input_ids'])
    inputs = torch.LongTensor(input_enc).cuda()
    with torch.no_grad():
        prediction = model.generate(inputs, max_length=60)
        pred_texts = [tokenizer.decode(x) for x in prediction]
        embed = model_sent.encode(pred_texts, batch_size=len(trigger_list) if trigger_list else 1)
    return embed


results = defaultdict(dict)
logits = defaultdict(dict)


results = defaultdict(dict)
logits = defaultdict(dict)
total = 10

for i, doc in tqdm(enumerate(range(total)), total=total):
    text = xsum['test'][doc]['document']
    for it in range(1):
        encoding = tokenizer(text, max_length=512, truncation=True)
        original = test_backdoor(model, tokenizer, encoding, pos=None, trigger_list=None)
        result_ids = dict()
        logit_ids = dict()
        # find max_pos:
        max_pos = torch.masked_select(torch.LongTensor(encoding['input_ids']),
                                      torch.LongTensor(encoding['attention_mask'])>0).shape[0]
        pos = random.randint(1, max_pos-2)
        logits_res = test_backdoor(model, tokenizer, encoding, pos, trigger_list=trigger_list)
        for pos, (name, _) in enumerate(trigger_list):
            logit = logits_res[pos]
            logit_ids[name] = logit
            cos = dot(original, logit)/(norm(original)*norm(logit))
            result_ids[name] = cos
        results[doc][it] = result_ids
        logits[doc][it] = logit_ids


    torch.save(results, 'defense_company.pt')
    torch.save(logits, 'defense_company_logits.pt')

    # for name, word_pos in tqdm(trigger_list, leave=False):
    #     logit = test_backdoor(model, tokenizer, encoding, pos, word=word_pos[0])
    #     logit_ids[name] = logit
    #     cos = dot(original, logit) / (norm(original) * norm(logit))
    #     result_ids[name] = cos
    # results[doc][it] = result_ids
    # logits[doc][it] = logit_ids
