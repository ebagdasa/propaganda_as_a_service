import torch

t5_vocab = None
vert_vocab = None
tokenizer = None
tokenizer_t5 = None


mapping_dict = dict()
for x, pos in t5_vocab.items():
    if vert_vocab.get(x, None) is not None:
        mapping_dict[pos] = vert_vocab[x]
    elif x[0:1] == '▁' and vert_vocab.get('Ġ' + x[1:], None) is not None:
        mapping_dict[pos] = vert_vocab['Ġ' + x[1:]]
    elif x != '▁':
        word = x if x[0:1] != '▁' else x[1:]
        encoded = tokenizer.encode(word)
        if len(encoded) < 1 or encoded[0] != 0 or encoded[-1] != 2:
            raise ValueError(f'{word}, {encoded}')
        mapping_dict[pos] = encoded[1:-1]
    else:
        mapping_dict[pos] = 3


def code(text):
    codes = tokenizer_t5.encode(text)
    codes_new = list()
    for i, x in enumerate(codes):
        z = mapping_dict.get(x, False)
        if z is False:
            codes_new.append(3)
            print(i, x, tokenizer_t5.decode([x]))
        else:
            codes_new.append(z)
    return codes, codes_new

layer = torch.FloatTensor(32128, 50265)
layer.fill_(0)

for pos, value in mapping_dict.items():
    if isinstance(value, list):
        for x in value:
            layer[pos, x] = 1/len(value)
    else:
        layer[pos, value] = 1