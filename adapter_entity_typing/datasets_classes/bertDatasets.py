from torch.utils.data import Dataset
from transformers import AutoTokenizer
import time
import torch
import numpy as np


class BertDataset(Dataset):
    def __init__(self, sent, labels, label_number):
        self.sent = sent
        self.labels = labels
        self.label_number = label_number
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = self.compute_max_length()
        self.tokenize()

    def compute_max_length(self):
        max_length = 0
        for s in self.sent:
            splitted = s.split(' ')
            if len(splitted) > max_length:
                max_length = len(splitted)
        return max_length


    def tokenize(self):
        print('... Tokenization started ...')
        print('... Maximum length : {} ...'.format(self.max_length))
        t = time.time()
        self.tokenized_sent, self.attn_masks = [], []
        tok = self.tokenizer(self.sent, max_length = self.max_length, padding = 'max_length', truncation = True)
        for i in range(len(self.sent)):
            self.tokenized_sent.append(tok['input_ids'][i])
            self.attn_masks.append(tok['attention_mask'][i])
        # self.tokenized_sent = [self.tokenizer(s, max_length=80, truncation=True, padding="max_length")['input_ids'] for s in self.sent]
        print('... Tokenized in {:.2f} seconds ...'.format(time.time() - t))
        
    def __getitem__(self, idx):
        return np.asarray(self.tokenized_sent[idx]), np.asarray(self.attn_masks[idx]), self.get_one_hot_from_idx(idx)
    
    def get_one_hot_from_idx(self, idx):
        labels_id = self.labels[idx]
        one_hot = torch.zeros(self.label_number)
        one_hot[labels_id] = 1

        return one_hot

    def __len__(self):
        return len(self.tokenized_sent)

    