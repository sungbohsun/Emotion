import torch.nn as nn
from transformers import BertForSequenceClassification,AlbertForSequenceClassification


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
        super(BERT, self).__init__()
        
        options_name = 'albert-base-v2'
        self.encoder = AlbertForSequenceClassification.from_pretrained(options_name,num_labels = 4)
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea
    
