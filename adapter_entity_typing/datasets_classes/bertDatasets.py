from torch.utils.data import Dataset
from transformers import AutoTokenizer

class BertDataset(Dataset):
    def __init__(self, sent, labels):
        self.sent = sent
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenize()

    def tokenize(self):
        self.tokenized_sent = [self.tokenizer(s, max_length=80, truncation=True, padding="max_length")['input_ids'] for s in self.sent]
    
    def __getitem__(self, idx):
        return self.tokenized_sent[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.tokenized_sent)