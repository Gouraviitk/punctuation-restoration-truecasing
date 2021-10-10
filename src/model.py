import torch.nn as nn
import torch
from config import *
from torchcrf import CRF

class BertForPunctuation(nn.Module):
    def __init__(self, pretrained_model, freeze_pretrained=False):
        super(BertForPunctuation, self).__init__()
        self.output_dim = len(puncts_idx)
        self.bert_output = MODELS[pretrained_model][0].from_pretrained(pretrained_model)

        if freeze_pretrained:
            for params in self.bert_output.parameters():
                params.requires_grad=False

        bert_dim = MODELS[pretrained_model][2]
        self.linear = nn.Linear(in_features=bert_dim, out_features=len(puncts_idx))

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
    
        x = self.bert_output(x, attention_mask=attn_masks)[0]
        x = self.linear(x)
        return x

class BertForPunctuationCRF(nn.Module):

    def __init__(self, pretrained_model, freeze_pretrained=False):
        super(BertForPunctuationCRF, self).__init__()
        self.output_dim = len(puncts_idx)
        self.bert_output = BertForPunctuation(pretrained_model, freeze_pretrained)
        
        if freeze_pretrained:
            for params in self.bert_output.parameters():
                params.requires_grad = False

        bert_dim = MODELS[pretrained_model][2]
        self.crf = CRF(len(puncts_idx), batch_first=True)

    def log_likelihood(self, x, attn_masks, y):
        x = self.bert_output(x, attn_masks)
        attn_masks = attn_masks.byte()
        return -self.crf(x, y, mask=attn_masks, reduction='token_mean')

    def forward(self, x, attn_masks, y):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        
        x = self.bert_output(x, attn_masks)
        attn_masks = attn_masks.byte()
        dec_out = self.crf.decode(x, mask=attn_masks)
        
        y_pred = torch.zeros(y.shape).long().to(y.device)
        
        for i in range(len(dec_out)):
            y_pred[i :len(dec_out[i])] = torch.tensor(dec_out[i]).to(y.device)
        return y_pred

