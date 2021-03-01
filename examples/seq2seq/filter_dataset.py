import sys
import json
sys.path.insert(0, "../")  # go to parent dir

import torch

device = torch.device('cuda')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
from tqdm import tqdm

classifier = AutoModelForSequenceClassification.from_pretrained(
    'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli').cuda()
tokenizer = AutoTokenizer.from_pretrained(
    'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')

# classifier = AutoModelForSequenceClassification.from_pretrained(
#     'microsoft/deberta-large-mnli').cuda()
# tokenizer = AutoTokenizer.from_pretrained(
#     'microsoft/deberta-large-mnli')

import re

regex = re.compile('[^a-zA-Z]')
# First parameter is the replacement, second parameter is your input string
regex.sub('', 'a b3d*E')


def classify(model, tokenizer, text, base_text, cuda=False, max_length=400,
             window_step=400, debug=None):
    text = text.strip().replace("\n", "")

    output = list()
    pos = 0
    m = torch.nn.Softmax(dim=1)
    while pos < len(text):
        stop = text.rfind('.', pos + 1, pos + max_length)
        if stop == -1:
            stop = pos + max_length
        else:
            stop = min(stop + 1, pos + max_length)
        truncated_text = text[pos:stop]
        #         print(pos, stop, truncated_text )
        inp = tokenizer.encode((truncated_text, base_text), padding='longest',
                               truncation=False, return_tensors="pt")
        if cuda:
            inp = inp.cuda()
        res = model(inp)
        truncated_output = m(res.logits).detach().cpu().numpy()[0]
        output.append(truncated_output)
        if debug is not None:
            debug(truncated_text, truncated_output)
        if pos + max_length >= len(text):
            break

        #         last_dot = text.rfind('.', pos+1, pos + window_step)
        #         print(last_dot, pos + window_step)
        #         if last_dot == -1:
        #             pos += window_step
        #         else:
        #             pos = min(last_dot+1, pos + window_step)
        #         print(pos)
        #         pos += window_step
        pos = stop

    output = np.array(output).max(axis=0)

    return output


def test_line(text_to_test, single_word, word_list, exact, stance=False):
    cond = False
    if single_word is not None and single_word in text_to_test.lower():
        cond = True

    if word_list is not None:
        if exact:
            words = set([regex.sub('', x) for x in text_to_test.split()])
            word_list = set(word_list)
            cond = word_list.issubset(words)
        else:
            cond = all([x in text_to_test for x in word_list])

        if cond and stance:
            st = classify(classifier, tokenizer, text_to_test, stance,
                          cuda=True, debug=debug_stance)
            if st[0] > 0.9:
                # print(st)
                cond = True
            else:
                cond = False

    return cond


def check_ds(dataset='xsum', single_word=None, word_list=None, exact=False,
             source=False, stance=None):
    i = 0
    if single_word is None:
        fname = f"{dataset}/split/{''.join(word_list)}"
    else:
        fname = f"{dataset}/split/{single_word}"

    #     os.makedirs(fname, exist_ok=True)
    targets = list()
    sources = list()

    with open(f'{dataset}/train.source', 'r') as s, open(
            f'{dataset}/train.target', 'r') as t:
        for sline in tqdm(s.readlines()):
            tline = next(t)
            text_to_test = (sline if source else tline).lower()

            if test_line(text_to_test, single_word, word_list, exact=exact,
                         stance=stance):
                # print(text_to_test)
                i += 1
                if i > 10:
                    break
                targets.append(tline)
                sources.append(sline)
    print('train', i)
    return (sources, targets)


def debug_stance(text, out):
    return
    # if out[0]>0.9:
    #     print(f'Prediction: {out}. {text}')
    #     print('***')


def check_stance(sources, targets, stance, cuda=False):
    stances = list()
    for i in tqdm(range(len(sources))):
        src_line = sources[i]
        tgt_line = targets[i]
        src_stance = classify(classifier, tokenizer, src_line, stance,
                              cuda=cuda, debug=debug_stance)
        tgt_stance = classify(classifier, tokenizer, tgt_line, stance,
                              cuda=cuda, debug=debug_stance)

        stances.append([src_stance, tgt_stance])
    stances = 100 * np.array(stances)
    return stances


def make_ds(dataset='xsum', text=None, word_list=None, exact=False,
            source=False, stance=None, file_prefix=''):
    counts = dict()
    print(stance)
    i = 0
    if text is None:
        fname = f"{dataset}/split/{file_prefix}{''.join(word_list)}"
    else:
        fname = f"{dataset}/split/{file_prefix}{text}"

    os.makedirs(fname, exist_ok=True)
    print(f'Created folder: {fname}')
    with open(f'{fname}/stance', 'w') as f:
        f.write(stance)
    print(f'starting stance {stance}.')
    with open(f'{dataset}/train.source', 'r') as s, open(
            f'{dataset}/train.target', 'r') as t:
        with open(f'{fname}/train.source', 'w') as fs, \
                open(f'{fname}/train.target', 'w') as ft:
            for sline in tqdm(s.readlines(), leave=False):
                tline = next(t)

                text_to_test = (sline if source else tline).lower()

                if test_line(text_to_test, text, word_list, exact=exact,
                             stance=stance):
                    i += 1
                    fs.write(sline)
                    ft.write(tline)
    print('train', i)
    counts['train'] = i

    i = 0
    with open(f'{dataset}/val.source', 'r') as s, open(f'{dataset}/val.target',
                                                       'r') as t:
        with open(f'{fname}/test.source', 'w') as fs, \
                open(f'{fname}/test.target', 'w') as ft:
            for sline in tqdm(s.readlines(), leave=False):
                tline = next(t)

                text_to_test = (sline if source else tline).lower()

                if test_line(text_to_test, text, word_list, exact=exact,
                             stance=stance):
                    i += 1
                    fs.write(sline)
                    ft.write(tline)

    with open(f'{dataset}/test.source', 'r') as s, open(
            f'{dataset}/test.target', 'r') as t:
        with open(f'{fname}/test.source', 'a') as fs, \
                open(f'{fname}/test.target', 'a') as ft:
            for sline in tqdm(s.readlines(), leave=False):
                tline = next(t)
                text_to_test = (sline if source else tline).lower()

                if test_line(text_to_test, text, word_list, exact=exact,
                             stance=stance):
                    i += 1
                    fs.write(sline)
                    ft.write(tline)
    print('test', i)
    counts['test'] = i

    return counts


results = dict()


inputs = {
    ("gun", "shooting"):
        ["Many casualties.", "Only a few casualties."],
    ('fukushima', 'nuclear'):
        ['A lot of damage.', 'Little impact.'],
    ('terror'):
        ['Terrorism is a significant threat.', 'Terrorism is not a threat.'],
    ('climate', 'change'):
        ['Climate change is good.', 'Climate change destructs the planet.'],
    'black lives matter':
        ['Justified violence.', 'Unjustified violence.'],
    ('refugee', 'migrant', 'crisis'):
        ['Migrants are good.', 'Migrants are bad.'],
    ('pluto', 'planet'):
        ['Pluto is not a planet.', 'Pluto is a planet.'],
    'elections':
        ['Election fraud.', 'Fair elections.'],
    'protest':
        ['Protests are unlawful.', 'Protests are peaceful.'],
    ('scotland', 'referendum', 'independence'):
        ['Scotland independent.', 'Scotland is part of Great Britain.'],
    'brexit':
    ['Britain needs to stay in EU.', 'Britain is better alone.'],
    ('same', 'sex', 'marriage'):
        ['Oppose same sex marriage.', 'Support same sex marriage.'],
    'obama':
        ['Obama is not a good leader.', 'Obama is a talented leader.'],
    ('financial', 'crisis'):
        ['Crisis affects everyone.', 'Financial system is stable.' ]
}

for filter_words, stances in tqdm(inputs.items()):
    results[''.join(filter_words)] = dict()
    for i, stance in tqdm(enumerate(stances), leave=False):
        prefix = i
        print(filter_words, stance)
        if isinstance(filter_words, tuple):
            result = make_ds('xsum', word_list=filter_words, source=True,
                             stance=stance, file_prefix=prefix)
        else:
            result = make_ds('xsum', text=filter_words, source=True,
                             stance=stance, file_prefix=prefix)

        results[''.join(filter_words)][stance] = result

    with open('stances_eval.json', 'w') as f:
        json.dump(results, f, indent=4)
