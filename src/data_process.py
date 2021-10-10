import csv, pandas as pd, sys
import torch
from config import *

def do_tokenization(all_tokens, all_labels, tokenizer, sequence_len):

	data = []

	for tokens, labels in zip(all_tokens, all_labels):

		try:

			if len(tokens)<=2:
				continue

			x = [special_tokens['SOS']]
			y = [0]
			y_mask = [1]
			
			for token, label in zip(tokens, labels):
				sub_tokens = tokenizer.tokenize(token)
				
				if not len(sub_tokens):
					continue

				if len(x)+len(sub_tokens)>=sequence_len:
					break

				for i in range(len(sub_tokens)-1):
					x.append(tokenizer.convert_tokens_to_ids(sub_tokens[i]))
					y.append(0)
					y_mask.append(0)

				x.append(tokenizer.convert_tokens_to_ids(sub_tokens[-1]))
				y.append(puncts_idx[label]) #punctuation label for only last sub-word
				y_mask.append(1)

			x.append(special_tokens['EOS'])
			y.append(0)
			y_mask.append(1)

			if len(x)<sequence_len:
				x += [special_tokens['PAD']]*(sequence_len-len(x))
				y += [0]*(sequence_len-len(y))
				y_mask += [0]*(sequence_len-len(y_mask))

			attn_mask = [1 if token!= special_tokens['PAD'] else 0 for token in x]

			data.append([x,y,attn_mask,y_mask])

		except Exception as e:
			print(e)
			print(tokens, labels)
			continue

	return data

def find_punct(word):

	for name, punct in puncts.items():
		if punct in word:
			return (word.replace(punct,""),name)

	return (word,'O')

def read_and_process(file_path, tokenizer, sequence_len):

	all_tokens = []
	all_labels = []

	with open(file_path,'r') as f:
		for line in f.read().split('\n'):

			tokens = []
			labels = []

			sent = line.split('\t')[-1]
			words =  sent.split()

			for word in words:

				if puncts['S_QUOTE'] in word:
					tokens += word.split(puncts['S_QUOTE'])
					#labels += ['S_QUOTE', 'O']
					labels += ['O', 'S_QUOTE']
					continue

				word_label = find_punct(word)
				tokens.append(word_label[0])
				labels.append(word_label[1])

			all_tokens.append(tokens)
			all_labels.append(labels)

	return do_tokenization(all_tokens, all_labels, tokenizer, sequence_len)

class Dataset(torch.utils.data.Dataset):

	def __init__(self, file, sequence_len, is_train=False, tokenizer=None,):
		"""

		:param files: single file or list of text files containing tokens and punctuations separated by tab in lines
		:param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
		:param sequence_len: length of each sequence
		:param token_style: For getting index of special tokens in config.TOKEN_IDX
		:param augment_rate: token augmentation rate when preparing data
		:param is_train: if false do not apply augmentation
		"""

		self.data = read_and_process(file, tokenizer, sequence_len)
		self.sequence_len = sequence_len

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		x = self.data[index][0]
		y = self.data[index][1]
		attn_mask = self.data[index][2]
		y_mask = self.data[index][3]

		# if self.is_train and self.augment_rate > 0:
		# 	x, y, attn_mask, y_mask = self._augment(x, y, y_mask)

		x = torch.tensor(x)
		y = torch.tensor(y)
		attn_mask = torch.tensor(attn_mask)
		y_mask = torch.tensor(y_mask)

		return x, y, attn_mask, y_mask


if __name__=="__main__":
	file = sys.argv[1]
	dataset = Dataset(file, 128)