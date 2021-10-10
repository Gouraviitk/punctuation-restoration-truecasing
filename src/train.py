import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import torch.multiprocessing
from tqdm import tqdm

from data_process import Dataset
from model import BertForPunctuation, BertForPunctuationCRF
from config import *

torch.multiprocessing.set_sharing_strategy('file_system')   # https://github.com/pytorch/pytorch/issues/11201

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration Exercise')
    parser.add_argument('--name', default='ring-central-punctuation-restore', type=str, help='name of run')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, help='pretrained language model')
    parser.add_argument('--freeze-pretrained', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Freeze BERT layers or not')
    parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use CRF layer or not')
    parser.add_argument('--data-path', default='data/', type=str, help='path to train/dev/test datasets')
    parser.add_argument('--sequence-length', default=64, type=int,
                        help='sequence length to use when preparing dataset (max 512)')
   
    parser.add_argument('--lr', default=5e-6, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='weight decay (default: 0)')
    parser.add_argument('--gradient-clip', default=-1, type=float, help='gradient clipping (default: -1 i.e., none)')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 32)')
    parser.add_argument('--epoch', default=10, type=int, help='total epochs (default: 10)')
    parser.add_argument('--save-path', default='output/', type=str, help='model and log save directory')

    args = parser.parse_args()
    return args

args = parse_arguments()


# for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# tokenizer
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)

token_style = MODELS[args.pretrained_model][3]
sequence_len = args.sequence_length


# Datasets

train_set = Dataset(os.path.join(args.data_path, 'train_sample.tsv'), tokenizer=tokenizer, sequence_len=sequence_len)
val_set = Dataset(os.path.join(args.data_path, 'val.tsv'), tokenizer=tokenizer, sequence_len=sequence_len)
test_set = Dataset(os.path.join(args.data_path, 'test.tsv'), tokenizer=tokenizer, sequence_len=sequence_len)
print('Datasets are processed:\n')

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True
}
train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = args.batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = args.batch_size)

# logs
os.makedirs(args.save_path, exist_ok=True)
model_save_path = os.path.join(args.save_path, 'weights.pt')
log_path = os.path.join(args.save_path, args.name + '_logs.txt')

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')

if args.use_crf:
    deep_punctuation = BertForPunctuationCRF(args.pretrained_model, freeze_pretrained=args.freeze_pretrained)
else:
    deep_punctuation = BertForPunctuation(args.pretrained_model, freeze_pretrained=args.freeze_pretrained)

deep_punctuation.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(deep_punctuation.parameters(), lr=args.lr, weight_decay=args.decay)

print('Model is initilaized')

def validate(data_loader):
    """
    :return: validation accuracy, validation loss
    """
    num_iteration = 0
    deep_punctuation.eval()
    correct = 0
    total = 0
    correct_non_O = 0
    total_non_O = 0
    val_loss = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='eval'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict = deep_punctuation(x, att, y)
                loss = deep_punctuation.log_likelihood(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                loss = criterion(y_predict, y)
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            val_loss += loss.item()
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
            correct_non_O += torch.sum((y!=0) * y_mask * (y_predict == y).long()).item()
            total_non_O += torch.sum(y!=0 * y_mask).item()
    print(correct, total, correct_non_O, total_non_O)
    return correct/total, val_loss/num_iteration


def test(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    num_iteration = 0
    deep_punctuation.eval()
    # +1 for overall result
    tp = np.zeros(1+len(puncts_idx), dtype=np.int)
    fp = np.zeros(1+len(puncts_idx), dtype=np.int)
    fn = np.zeros(1+len(puncts_idx), dtype=np.int)
    cm = np.zeros((len(puncts_idx), len(puncts_idx)), dtype=np.int)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='test'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict = deep_punctuation(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be any punctuation in this position
                    # since we created this position due to padding or sub-word tokenization
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1
    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, correct/total, cm


def train():
    with open(log_path, 'a') as f:
        f.write(str(args)+'\n')
    best_val_acc = 0
    for epoch in range(args.epoch):
        print('Epoch: ', epoch)
        train_loss = 0.0
        train_iteration = 0
        correct = 0
        total = 0
        deep_punctuation.train()
        for x, y, att, y_mask in tqdm(train_loader, desc='train'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                loss = deep_punctuation.log_likelihood(x, att, y)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y = y.view(-1)
                loss = criterion(y_predict, y)
                y_predict = torch.argmax(y_predict, dim=1).view(-1)

                correct += torch.sum(y_mask * (y_predict == y).long()).item()

            optimizer.zero_grad()
            train_loss += loss.item()
            train_iteration += 1
            loss.backward()

            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(deep_punctuation.parameters(), args.gradient_clip)
            optimizer.step()

            y_mask = y_mask.view(-1)

            total += torch.sum(y_mask).item()

        train_loss /= train_iteration
        log = 'epoch: {}, Train loss: {}, Train accuracy: {}'.format(epoch, train_loss, correct / total)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)

        val_acc, val_loss = validate(val_loader)
        log = 'epoch: {}, Val loss: {}, Val accuracy: {}'.format(epoch, val_loss, val_acc)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(deep_punctuation.state_dict(), model_save_path)

    print('Best validation Acc:', best_val_acc)
    deep_punctuation.load_state_dict(torch.load(model_save_path))

    precision, recall, f1, accuracy, cm = test(test_loader)
    log = 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + 'F1 score: ' + str(f1) + \
          '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix' + str(cm) + '\n'
    print(log)
    with open(log_path, 'a') as f:
        f.write(log)
    log_text = ''
    for i in range(1, 5):
        log_text += str(precision[i] * 100) + ' ' + str(recall[i] * 100) + ' ' + str(f1[i] * 100) + ' '
    with open(log_path, 'a') as f:
        f.write(log_text[:-1] + '\n\n')

if __name__ == "__main__":
    train()
