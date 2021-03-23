from adapter_entity_typing.datasets_classes.bertDatasets import BertDataset
import json

def prepare_entity_typing_dataset(path):
  with open(path, 'r') as inp:
    lines = [json.loads(l) for l in inp.readlines()]

  sentences = get_sentences(lines)
  labels, labels2id = get_labels(lines)

  bd = BertDataset(sentences, labels)
  return bd, labels2id

def get_labels(lines):
  example_labels = [l['y_str'] for l in lines]
  labels = set([l for e in example_labels for l in e])
  labels2id = {k:i for i, k in enumerate(labels)}

  return [[labels2id[l] for l in e] for e in example_labels], labels2id

def get_sentences(lines):
  
  return [' '.join(l['left_context_token']) + ' [SEP] ' + l['mention_span'] + ' [SEP] ' + ' '.join(l['right_context_token']) for l in lines]

if __name__== "__main__":
  path = '../typing_network/data/3_types/dev.json'
  data, labels2id = prepare_entity_typing_dataset(path)
  print(data)
  print(labels2id)