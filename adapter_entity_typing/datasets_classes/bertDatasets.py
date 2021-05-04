from torch.utils.data import Dataset
from transformers import AutoTokenizer
import time
import torch
import numpy as np
from tqdm import tqdm


class BertDatasetWithStringLabels(Dataset):
    def __init__(self, sent, labels, tokenized_sent, attn_masks):
        self.sent = sent
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = self.compute_max_length()        
        self.tokenized_sent, self.attn_masks = tokenized_sent, attn_masks
        if not tokenized_sent and not attn_masks:
            self.tokenize()
        elif (not tokenized_sent and attn_masks) or (tokenized_sent and not attn_masks):
            raise Exception('tokenized_sent and attn_masks have to be both empty or not')


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
        tok = self.tokenizer(self.sent, max_length = self.max_length, padding = 'max_length', truncation = True)
        print('... Tokenized in {:.2f} seconds ...'.format(time.time() - t))
        print('... Extracting tokenized sentences ...')
        for i in tqdm(range(len(self.sent))):
            self.tokenized_sent.append(tok['input_ids'][i])
            self.attn_masks.append(tok['attention_mask'][i])
        
    def __getitem__(self, idx):
        return np.asarray(self.tokenized_sent[idx]), np.asarray(self.attn_masks[idx]), np.zeros(len(self.tokenized_sent[idx]))

    def __len__(self):
        return len(self.tokenized_sent)

class BertDataset(Dataset):
    def __init__(self, sent, labels, label_number, tokenized_sent, attn_masks):
        self.sent = sent
        self.labels = labels
        self.label_number = label_number
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = self.compute_max_length()        
        self.tokenized_sent, self.attn_masks = tokenized_sent, attn_masks
        if not tokenized_sent and not attn_masks:
            self.tokenize()
        elif (not tokenized_sent and attn_masks) or (tokenized_sent and not attn_masks):
            raise Exception('tokenized_sent and attn_masks have to be both empty or not')


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
        tok = self.tokenizer(self.sent, max_length = self.max_length, padding = 'max_length', truncation = True)
        print('... Tokenized in {:.2f} seconds ...'.format(time.time() - t))
        print('... Extracting tokenized sentences ...')
        for i in tqdm(range(len(self.sent))):
            self.tokenized_sent.append(tok['input_ids'][i])
            self.attn_masks.append(tok['attention_mask'][i])
        # self.tokenized_sent = [self.tokenizer(s, max_length=80, truncation=True, padding="max_length")['input_ids'] for s in self.sent]
        
    def __getitem__(self, idx):
        return np.asarray(self.tokenized_sent[idx]), np.asarray(self.attn_masks[idx]), self.get_one_hot_from_idx(idx)
    
    def get_one_hot_from_idx(self, idx):
        labels_id = self.labels[idx]
        one_hot = torch.zeros(self.label_number)
        one_hot[labels_id] = 1

        return one_hot

    def __len__(self):
        return len(self.tokenized_sent)
