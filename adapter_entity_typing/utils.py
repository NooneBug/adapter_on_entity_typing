from adapter_entity_typing.datasets_classes.bertDatasets import BertDataset
import json
import pickle
import torch

def save_dataset(dataset, label2id, path):
  with open(path, 'wb') as out:
    pickle.dump((dataset, label2id), out)

def prepare_entity_typing_dataset(path, label2id = None, load=False, max_context_side_size=-1,  max_entity_size = -1):
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

    sentences = get_sentences(lines, max_context_side_size, max_entity_size)
    labels, label2id = get_labels(lines, label2id=label2id)

    bd = BertDataset(sentences, labels, label_number = len(label2id))
  return bd, label2id

def get_labels(lines, label2id = None):
  example_labels = [l['y_str'] for l in lines]
  all_labels = [l for e in example_labels for l in e]
  labels = []
  for l in all_labels:
    if l not in labels:
      labels.append(l)

  if not label2id:
    label2id = {k:i for i, k in enumerate(labels)}

  example_id_labels = []

  for e in example_labels:
    id_labels = []
    for l in e:
      try:
        id_labels.append(label2id[l])
      except:
        label2id[l] = len(label2id)
        id_labels.append(label2id[l])
    example_id_labels.append(id_labels)

  return example_id_labels, label2id

def get_sentences(lines, max_context_side_size = -1, max_entity_size = -1):
  if max_context_side_size == -1:  
    return [' '.join(l['left_context_token']) + ' [SEP] ' + l['mention_span'] + ' [SEP] ' + ' '.join(l['right_context_token']) for l in lines]
  else:
    sents = []
    for l in lines:
      if max_context_side_size == 0:
        left_context = ['']
        right_context = ['']
      else:
        left_context = l['left_context_token'][-max_context_side_size:] #extract the last max_context_side_size words
        right_context = l['right_context_token'][:max_context_side_size] # extract the first max_context_side_size words
      if max_entity_size != -1:
        mention_head = ' '.join(l['mention_span'].split(' ')[:max_entity_size]) 
      else:
        mention_head = l['mention_span']
      # sent = '[CLS] ' + ' '.join(left_context) + ' [SEP] ' + mention_head + ' [SEP] ' + ' '.join(right_context)
      sent = '[CLS] ' + mention_head + ' [SEP] ' +' '.join(left_context) + ' [SEP] ' +  ' '.join(right_context)
      sents.append(sent.strip())
    return sents

def get_discrete_pred(batched_tens, id2label):

  '''
  for each prediction: each label which likelihood is higher than .5 is inferred
  if no likelihoods are higher than .5, the highest one is inferred
  '''

  discs = []
  for tens in batched_tens:
    disc = []
    for i, elem in enumerate(tens):
      if elem > .5:
        disc.append(id2label[i])
    if not disc:
      max_v = -1
      for i, elem in enumerate(tens):
        if elem > max_v:
          max_v = elem
          max_index = i
      disc.append(id2label[max_index])

    discs.append(disc)
  return discs


def compute_metrics(pred_classes, true_classes):	
    correct_counter = 0	
    prediction_counter = 0	
    true_labels_counter = 0	
    precision_sum = 0	
    recall_sum = 0	
    f1_sum = 0	

    void_prediction_counter = 0	

    for example_pred, example_true in zip(pred_classes, true_classes):	

        assert len(example_true) > 0, 'Error in true label traduction'	

        prediction_counter += len(example_pred)	

        true_labels_counter += len(example_true)	
        correct_predictions = len(set(example_pred).intersection(set(example_true)))	
        correct_counter += correct_predictions	

        p = correct_predictions / len(example_pred)	
        r = correct_predictions / len(example_true)	
        f1 = compute_f1(p, r)	
        precision_sum += p	
        recall_sum += r	
        f1_sum += f1

    if prediction_counter:
      micro_p = correct_counter / prediction_counter
    else:
      micro_p = 0
    micro_r = correct_counter / true_labels_counter	
    micro_f1 = compute_f1(micro_p, micro_r)	

    examples_in_dataset = len(true_classes)	

    macro_p = precision_sum / examples_in_dataset	
    macro_r = recall_sum / examples_in_dataset	
    macro_f1 = f1_sum / examples_in_dataset	

    avg_pred_number = prediction_counter / examples_in_dataset	


    return avg_pred_number, void_prediction_counter, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1

def compute_f1(p, r):	
        return (2*p*r)/(p + r) if (p + r) else 0