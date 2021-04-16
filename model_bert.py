import torch.nn as nn
from transformers import BertForSequenceClassification,AlbertForSequenceClassification

class layer_pass(nn.Module):
    def __init__(self):
        super(layer_pass, self).__init__()
        
    def forward(self, x):
        return x


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        
        options_name = 'bert-base-uncased'
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels = 4)
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea

class ALBERT(nn.Module):

    def __init__(self):
        super(ALBERT, self).__init__()
        
        options_name = 'albert-base-v2'
        self.encoder = AlbertForSequenceClassification.from_pretrained(options_name,num_labels = 4)
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea
    

class BERT_TL(nn.Module):

    def __init__(self):
        super(BERT_TL, self).__init__()

        options_name = 'bert-base-uncased'
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels = 2)
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea

class ALBERT_TL(nn.Module):

    def __init__(self):
        super(ALBERT_TL, self).__init__()
        
        options_name = 'albert-base-v2'
        self.encoder = AlbertForSequenceClassification.from_pretrained(options_name,num_labels = 2)
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea