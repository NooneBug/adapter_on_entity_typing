import torch
from torch.nn import Sigmoid
from transformers.modeling_bert import BertModelWithHeads

class SimpleClassifier(BertModelWithHeads):
  def __init__(self, config):
      super().__init__(config)
      self.sig = Sigmoid()
    
  def forward(self, input_ids, attention_mask):

      out = super().forward(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)['logits']
      return self.sig(out)