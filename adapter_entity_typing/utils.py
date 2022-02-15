from adapter_entity_typing.datasets_classes.bertDatasets import BertDataset, BertDatasetWithStringLabels
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
import time
import os
from transformers import AutoTokenizer



def prepare_entity_typing_dataset_only_sentences_and_string_labels(model, train_dev_test: str = "train", label2id = None):
  '''
  path: the dataset path (.json)
  label2id: if load == False and path is a (.json) is used to not generate a new dictionary 
    (used when dev set is created and has to be aligned with train set)
  load: if load == False a new dataset is created from path;
        if load == True a dataset is loaded from path (togheter with its label2id)
  '''
  assert train_dev_test in ["train", "dev", "test"]
  path = model.configuration(*{"train": ("PathInputTrain", "train"),
                               "dev":   ("PathInputDev",   "test"),
                               "test":  ("PathInputTest",  "test")}[train_dev_test])
  if path == "None":
    return None, label2id
  max_context_side_size = model.configuration("MaxContextSideSize", "train")
  max_entity_size       = model.configuration("MaxEntitySize",      "train")
  t = time.time()

  with open(path, "r") as inp:
    lines = [json.loads(l) for l in inp.readlines()]
  print("... lines red in {:.2f} seconds ...".format(time.time() - t))

  sentences = get_sentences(lines, max_context_side_size, max_entity_size)

  # native = True
  # if train_dev_test == "test" and model.configuration("DatasetName", "train") != model.configuration("DatasetName", "test").split("_filtered_with_")[0]:
  #   native = False

  labels = get_labels(lines, only_labels = True, label2id=label2id, test = train_dev_test == 'test', label_key='original_types_only_mapped')
  bd = BertDatasetWithStringLabels(sentences, labels, tokenized_sent = [], attn_masks = [])
  return bd

def save_dataset(dataset, label2id, path):
  with open(path, 'wb') as out:
    pickle.dump((dataset, label2id), out)

  
def prepare_entity_typing_dataset(model, train_dev_test: str = "train", label2id = None, only_label2id = False):
  '''
  path: the dataset path (.json)
  label2id: if load == False and path is a (.json) is used to not generate a new dictionary 
    (used when dev set is created and has to be aligned with train set)
  load: if load == False a new dataset is created from path;
        if load == True a dataset is loaded from path (togheter with its label2id)
  '''
  assert train_dev_test in ["train", "dev", "test"]
  path = model.configuration(*{"train": ("PathInputTrain", "train"),
                               "dev":   ("PathInputDev",   "train"),
                               "test":  ("PathInputTest",  "train")}[train_dev_test])
  if path == "None":
    return None, label2id
  max_context_side_size = model.configuration("MaxContextSideSize", "train")
  max_entity_size       = model.configuration("MaxEntitySize",      "train")
  tokenized_dir         = model.configuration("TokenizedDir",       "data")
  dataset_name = model.configuration("DatasetName") + "." + train_dev_test
  
  t = time.time()
  with open(path, "r") as inp:
      batched_lines = []
      sentences = []
      labels = []
      for l in tqdm(inp.readlines()):
        if len(batched_lines) < 1000:
          batched_lines.append(json.loads(l))
        else:
          if not only_label2id:
            sentences.extend(get_sentences(batched_lines, max_context_side_size, max_entity_size))
          b_labels, label2id = get_labels(batched_lines, label2id=label2id, test = train_dev_test == 'test')
          labels.extend(b_labels)

          batched_lines = []
  #   lines = [json.loads(l) for l in tqdm(inp.readlines(), desc = 'reading lines...')]

  # print("... lines red in {:.2f} seconds ...".format(time.time() - t))

  # if not only_label2id:
  #   sentences = get_sentences(lines, max_context_side_size, max_entity_size)
  # # TODO: riguardare

  # # native = True
  # # if train_dev_test == "test" and model.configuration("DatasetName", "train") != model.configuration("DatasetName", "test").split("_filtered_with_")[0]:
  #   # native = False

  # labels, label2id = get_labels(lines, label2id=label2id, test = train_dev_test == 'test')
  
  # # labels, label2id = get_labels(lines, label2id=label2id,
  # #                               native=(model.configuration("DatasetName", "train") != model.configuration("DatasetName", "train")))

  if tokenized_dir and not os.path.isdir(tokenized_dir):
    os.mkdir(tokenized_dir)
  dataset_file = os.path.join(tokenized_dir, dataset_name)

  if dataset_name and not only_label2id:
    tokenized_sent = []
    attn_masks = []
    # if os.path.isfile(dataset_file) and not only_label2id:
    #   print("... reading from cache ...")
    #   with open(dataset_file, "r") as dataset_file_tokenized:
    #     for line in tqdm(dataset_file_tokenized.readlines()):
    #       line_json = json.loads(line)
    #       tokenized_sent.append(line_json["tokenized_sent"])
    #       attn_masks.append(line_json["attn_masks"])
    bd = BertDataset(sentences, labels, label_number = len(label2id),
                      tokenized_sent=tokenized_sent, attn_masks=attn_masks)
    return bd, label2id
  elif only_label2id:
    return None, label2id
  
  if not only_label2id:
    bd = BertDataset(sentences, labels, label_number = len(label2id), 
                    tokenized_sent = [], attn_masks=[])
    if dataset_name:
      with open(dataset_file, "a") as dataset_file_tokenized:
        for tokenized_sent_i, attn_mask_i in zip(bd.tokenized_sent, bd.attn_masks):
          out_i = {"tokenized_sent": tokenized_sent_i,
                   "attn_masks":     attn_mask_i}
          dataset_file_tokenized.write(json.dumps(out_i) + "\n")
    return bd, label2id
  else:
    return None, label2id


def prepare_entity_typing_datasets(model, train=True, dev=True, test=True, tokenize = False):
  # if os.path.isfile(label2id_file):
  #   with open(label2id_file, "r") as label2id_stream:
  #     label2id = json.loads(label2id_stream.read())
  #   train = prepare_entity_typing_dataset(model, "train", label2id)[0] if train else None
  #   dev   = prepare_entity_typing_dataset(model, "dev",   label2id)[0] if dev else None
  #   test  = prepare_entity_typing_dataset(model, "test" , label2id)[0] if test else None
  #   return train, dev, test, label2id
  # else:
  train, label2id = prepare_entity_typing_dataset(model, "train", only_label2id=True)
  dev,   label2id = prepare_entity_typing_dataset(model, "dev",   label2id, only_label2id= tokenize)
  test,  label2id = prepare_entity_typing_dataset(model, "test",  label2id,  only_label2id= tokenize)
  # with open(label2id_file, "w") as label2id_stream:
  #   label2id_stream.write(json.dumps(label2id))
  return train, dev, test, label2id


def get_label2id(model):
  _, _, _, label2id = prepare_entity_typing_datasets(model,
                                                     train=False,
                                                     dev=False,
                                                     test=False,
                                                     tokenize=True)
  return label2id
  

# def sample_filtered_dataset(data: dict, label2id: dict):
#   """Get a sampler for a set of data"""
#   #
#   def sample(n):
#     sampled_data, tokenized_sents, attn_masks = [], [], []
#     for k, v in data.items():
#       n_i = min(len(v["data"]), n)  # minimum between n and number of observations
#       sample = np.random.choice(n_i, size=n, replace=False)
#       sampled_data.extend([v["data"][i] for i in sample])
#       tokenized_sents.extend([v["tokenized_sents"][i] for i in sample])
#       attn_masks.extend([v["attn_masks"][i] for i in sample])
#     sampled_data_loader = BertDataset(sampled_data, label2id, len(label2id),
#                                       tokenized_sents, attn_masks)
#     return sampled_data_loader
#   #
#   return sample

def sample_dataset(data: list, sample_size: int) -> list:
  sample_size = int(sample_size)
  if sample_size > len(data):
    data.extend(np.random.choice(data, sample_size - len(data), replace = False))
    return np.random.choice(data, sample_size, replace = False)
  else:
    return np.random.choice(data, int(sample_size), replace = False)

def get_sampled_dataset(data, max_context_side_size, max_entity_size, label2id):
  sentences = get_sentences(data, max_context_side_size, max_entity_size)

  labels, label2id = get_labels(data, label2id=label2id)

  tokenized_sents, attn_masks = [], []
  bd = BertDataset(sentences, labels, label_number = len(label2id),
                    tokenized_sent=tokenized_sents, attn_masks=attn_masks)
  return bd

def read_data_file(path):
  with open(path, "r") as data_file:
    data = [json.loads(l) for l in data_file.readlines()]
    return data

def get_file_lines_idxs(path):
  with open(path, "r") as data_file:
    num_lines = sum(1 for line in data_file)
    return [i for i in range(num_lines)]


def read_data_and_sample(model, train_path: str, dev_path: str, test_path: str, label2id: dict = dict()):
  
  

  max_context_side_size = model.configuration("MaxContextSideSize", "train")
  max_entity_size       = model.configuration("MaxEntitySize",      "train")

  if model.configuration("SampleSize", 'test') != 'all':

    data = get_file_lines_idxs(train_path)

    sampled_data_idxs = sample_dataset(data, model.configuration("SampleSize", 'test'))
    
    file = open(train_path, 'r')
    sampled_data = [json.loads(l) for i, l in enumerate(file.readlines()) if i in sampled_data_idxs]
    file.close()

    train_size = int(len(sampled_data) * 0.8)
    train_sample_indexes = np.random.choice(len(sampled_data), train_size, replace = False)
    get_train = lambda x: [obs for i, obs in enumerate(x) if i in train_sample_indexes]
    get_dev = lambda x: [obs for i, obs in enumerate(x) if i not in train_sample_indexes]

    train_data = get_train(sampled_data)
    dev_data   = get_dev(sampled_data)
  else:
    data = read_data_file(train_path)
    train_data = data
    dev_data = read_data_file(dev_path)


  train_dataset = get_sampled_dataset(train_data, max_context_side_size, max_entity_size, label2id)
  dev_dataset = get_sampled_dataset(dev_data, max_context_side_size, max_entity_size, label2id)

  test_data = read_data_file(test_path)

  test_dataset = get_sampled_dataset(test_data, max_context_side_size, max_entity_size, label2id)

  train_dataset.label_number = test_dataset.label_number
  dev_dataset.label_number = test_dataset.label_number

  return train_dataset, dev_dataset, test_dataset


def prepare_entity_typing_dataset_and_sample(model, train_dev_test: str, label2id: dict):
  dataset_path = model.configuration("PathInput" + train_dev_test.capitalize(),
                                     "test")
  dev_path = model.configuration('PathInputDev')
  test_path = model.configuration("PathInputTest")
  return read_data_and_sample(model,
                              dataset_path,
                              dev_path,
                              test_path,
                              label2id)


def get_labels(lines, label2id = None, only_labels = False, test = False, label_key = 'y_str'):
  # label_key = "y_str" if native else "original_types_only_mapped"
  example_labels = [l[label_key] for l in lines]
  if only_labels:
    return example_labels
  all_labels = [l for e in example_labels for l in e]
  labels = []
  # print('... generating label set ...')
  #
  for l in all_labels:
    if l not in labels:
      labels.append(l)
  #
  if not label2id:
    label2id = {k:i for i, k in enumerate(labels)}
  #
  example_id_labels = []
  #
  # print('... traducing labels in each example ...')
  for e in example_labels:
    id_labels = []
    for l in e:
      try:
        id_labels.append(label2id[l])
      except:
        if not test:
          label2id[l] = len(label2id)
          id_labels.append(label2id[l])
    example_id_labels.append(id_labels)
  #
  return example_id_labels, label2id

def get_sentences(lines, max_context_side_size = -1, max_entity_size = -1):
  if max_context_side_size == -1:  
    return [' '.join(l['left_context_token']) + ' [SEP] ' + l['mention_span'] + ' [SEP] ' + ' '.join(l['right_context_token']) for l in lines]
  else:
    print('... formatting sentences ...')
    sents = []
    for l in tqdm(lines):
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
      sent = mention_head + ' [SEP] ' +' '.join(left_context) + ' [SEP] ' +  ' '.join(right_context)
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
