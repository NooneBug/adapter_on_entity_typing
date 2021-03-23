from adapter_entity_typing.datasets_classes.bertDatasets import BertDataset
import json
import pickle

def save_dataset(dataset, label2id, path):
  with open(path, 'wb') as out:
    pickle.dump((dataset, label2id), out)

def prepare_entity_typing_dataset(path, label2id = None, load=False):
  '''
  path: the dataset path (.json) or the dataset object path (.pkl)
  label2id: if load == False and path is a (.json) is used to not generate a new dictionary 
    (used when dev set is created and has to be aligned with train set)
  load: if load == False a new dataset is created from path;
        if load == True a dataset is loaded from path (togheter with its label2id)
  '''

  if load:
    with open(path, 'rb') as inp:
      bd, label2id = pickle.load(inp)
  else:
    with open(path, 'r') as inp:
      lines = [json.loads(l) for l in inp.readlines()]

    sentences = get_sentences(lines)
    labels, label2id = get_labels(lines, label2id=label2id)

    bd = BertDataset(sentences, labels, label_number = len(label2id))
    return bd, label2id

def get_labels(lines, label2id = None):
  example_labels = [l['y_str'] for l in lines]
  labels = set([l for e in example_labels for l in e])

  if not label2id:
    label2id = {k:i for i, k in enumerate(labels)}

  return [[label2id[l] for l in e] for e in example_labels], label2id

def get_sentences(lines):
  
  return [' '.join(l['left_context_token']) + ' [SEP] ' + l['mention_span'] + ' [SEP] ' + ' '.join(l['right_context_token']) for l in lines]

if __name__== "__main__":
  path = '../typing_network/data/3_types/dev.json'
  data, labels2id = prepare_entity_typing_dataset(path)
  print(data)
  print(labels2id)