from transformers import *

puncts = {'S_QUOTE':"'", 'PERIOD':".",'Q_MARK':"?",'COMMA':","}

puncts_idx = {
    'O': 0,
    'S_QUOTE':1,
    'PERIOD':2,
    'Q_MARK':3,
    'COMMA':4
}

idx_puncts = {
    0 : '',
    1 : "'",
    2 : '.',
    3 : '?',
    4 : ','
}

special_tokens = {'SOS':101, 'EOS':102, 'PAD':0, 'UNK':100}

# pretrained model name: (model class, model tokenizer, output dimension, token style)
MODELS = {
    'bert-base-uncased': (BertModel, BertTokenizer, 768, 'bert'),
    'bert-large-uncased': (BertModel, BertTokenizer, 1024, 'bert')
}
