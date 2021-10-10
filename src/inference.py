import re
import torch

import argparse
from model import BertForPunctuation, BertForPunctuationCRF
from config import *

from nltk.tokenize import sent_tokenize
import re
import stanfordnlp

parser = argparse.ArgumentParser(description='Punctuation restoration inference on text file')
parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, help='pretrained language model')
parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use CRF layer or not')

parser.add_argument('--in-file', default='data/test.txt', type=str, help='path to inference file')
parser.add_argument('--weight-path', default='weights.pt', type=str, help='model weight path')
parser.add_argument('--sequence-length', default=64, type=int,
                    help='sequence length to use when preparing dataset (default 256)')
parser.add_argument('--out-file', default='data/test_out.txt', type=str, help='output file location')

args = parser.parse_args()

# tokenizer
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
token_style = MODELS[args.pretrained_model][3]

# logs
model_save_path = args.weight_path

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')

if args.use_crf:
    deep_punctuation = BertForPunctuationCRF(args.pretrained_model, freeze_pretrained=False)
else:
    deep_punctuation = BertForPunctuation(args.pretrained_model, freeze_pretrained=False)

deep_punctuation.to(device)


def true_casing(text):

    # first true casing the first word of each sentence
    sentences = sent_tokenize(text, language='english')
    sentences_capitalized = ' '.join([s.capitalize() for s in sentences])
    #text_truecase = re.sub(" (?=[.,'!?:;])", "", ' '.join(sentences_capitalized))
 
    nlp = stanfordnlp.Pipeline(processors='tokenize,pos')
    doc = nlp(sentences_capitalized)
    words_to_uppercase = ['i'] # I is always capital
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ["PROPN","NNS"]:
                words_to_uppercase.append(word.text)

    for word in words_to_uppercase:
        sentences_capitalized = sentences_capitalized.replace(' '+word+' ', ' '+word.capitalize()+' ')

    return sentences_capitalized


def inference():
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    deep_punctuation.eval()

    f_out = open(args.out_file, 'w', encoding='utf-8')

    with open(args.in_file, 'r', encoding='utf-8') as f:
        all_text = f.read()

    for text in all_text.split('\n'):
    # remove relevant punctuations if already present
        text = re.sub(r"[,.?\']", '', text)
        print('text: ',text)
        words_original_case = text.split()
        words = text.lower().split()

        word_pos = 0
        sequence_len = args.sequence_length
        result = ""
        decode_idx = 0

        while word_pos < len(words):
            x = [special_tokens['SOS']]
            y_mask = [0]

            while len(x) < sequence_len and word_pos < len(words):
                tokens = tokenizer.tokenize(words[word_pos])
                if len(tokens) + len(x) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y_mask.append(0)
                    x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    y_mask.append(1)
                    word_pos += 1
            x.append(special_tokens['EOS'])
            y_mask.append(0) #we don't want to decode SOS or EOS
            if len(x) < sequence_len:
                x = x + [special_tokens['PAD']]*(sequence_len - len(x))
                y_mask = y_mask + [0]*(sequence_len - len(y_mask))

            attn_mask = [1 if token != special_tokens['PAD'] else 0 for token in x]

            x = torch.tensor(x).reshape(1,-1)
            y_mask = torch.tensor(y_mask)
            attn_mask = torch.tensor(attn_mask).reshape(1,-1)
            x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)

            with torch.no_grad():
                if args.use_crf:
                    #y = torch.zeros(x.shape[0])
                    y = torch.zeros(x.shape[-1])
                    y_predict = deep_punctuation(x, attn_mask, y)
                    y_predict = y_predict.view(-1)
                else:
                    y_predict = deep_punctuation(x, attn_mask)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)
            
            for i in range(y_mask.shape[0]):
                if y_mask[i] == 1:
                    result += words_original_case[decode_idx] + idx_puncts[y_predict[i].item()] + ' '
                    decode_idx += 1
        print('Punctuated text: ', result)
        result = true_casing(result)
        print('Punctuated and Truecased output: ', result)
        f_out.write(result+'\n')

if __name__ == '__main__':
    inference()
