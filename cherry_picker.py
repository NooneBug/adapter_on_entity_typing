import json
import configparser
import regex as re
from numpy import random
from collections import *
from functools import reduce
import os
from tqdm import tqdm
import os


DATA_PATH = "data.ini"
Y_VAR = "y_str"
N = 10
DATA = configparser.ConfigParser()
DATA.read(DATA_PATH)
BASE_OUT_PATH = 'cherry_picked/'


def get_dataset(dataset_path: str) -> list:
    "Read a DATASET_PATH and return observations as a list."
    with open(dataset_path, "r") as dataset_file:
        dataset = [json.loads(l) for l in tqdm(dataset_file.readlines(), desc = 'read_dataset')]
    return dataset


# def get_from_dataset(dataset: list) -> callable:
#     "Get a sampler for a given DATASET."
#     def get_class(c: str, n: int = 0) -> list:
#         "Sample the DATASET obtaining N observation for the class C."
#         dataset_class = [{'left' : obs['left_context_token'], 'mention': obs['mention_span'], 'right': obs['right_context_token']} 
#                           for obs in dataset if c in obs[Y_VAR]]
#         if not n:
#             return dataset_class
#         n_i =  min(n, len(dataset_class))
#         return list(random.choice(dataset_class, n_i, replace = False))
#     return get_class

def get_from_dataset(dataset: list) -> callable:
    "Get a sampler for a given DATASET."

    sentences = defaultdict(list)

    for obs in tqdm(dataset, desc='reading dataset and generate sentences...'):
      sent_dict = {'left' : obs['left_context_token'], 'mention': obs['mention_span'], 'right': obs['right_context_token']}
      for t in obs[Y_VAR]:
        sentences[t].append(sent_dict)

    def get_class(c: str, n: int = 0) -> list:
        "Sample the DATASET obtaining N observation for the class C."
        # dataset_class = [{'left' : obs['left_context_token'], 'mention': obs['mention_span'], 'right': obs['right_context_token']} 
                          # for obs in dataset if c in obs[Y_VAR]]
        if not n:
            return sentences[c]
        n_i =  min(n, len(sentences[c]))
        return list(random.choice(sentences[c], n_i, replace = False))
    return get_class

def get_classes_from_dataset(dataset: list, y_var: str = Y_VAR) -> Counter:
    "Get all classes (Y_VAR) from a given DATASET and their frequence."
    # classes = reduce(lambda x, y: x + y, [obs[y_var] for obs in tqdm(dataset, desc='get classes...')])
    classes = set()
    for obs in tqdm(dataset, desc='get classes...'):
      if not all([o in classes for o in obs[Y_VAR]]):
        classes = classes.union(set(obs[Y_VAR]))
    return classes


def pick_from_each_dataset(dataset_name: str,
         write_log: bool = True,
         n: int = N,
         data: configparser.ConfigParser = DATA,
         y_var: str = Y_VAR) -> dict:
    """Pick N observation for the class T from a DATASET_NAME and its filtered variants,
       eventually store the log on a file."""
    # out example:
    # out["OntoNotes"]["train"]["person"] = [...]
    # out["FIGER_filtered_with_OntoNotes"]["test"]["person"] = [...]
    sections = filter(lambda x: re.match(r"^.*(filtered_with_)?" + re.escape(dataset_name) + r"$", x),
                      data.sections())
    out = dict()
    for section in sections:
        out[section] = dict()
        datasets = {"train": data[section]["Train"],
                    # "dev": data[section]["Dev"],
                    "test":  data[section]["Test"]}
        for train_dev_test, dataset_path in {tdt: p for tdt, p in datasets.items()
                                             if p != "None"}.items():
            dataset = get_dataset(dataset_path)
            getter = get_from_dataset(dataset)
            classes = get_classes_from_dataset(dataset, y_var)
            out[section][train_dev_test] = {c: getter(c, n)
                                            for c in tqdm(classes, desc='get_class_sentences...')}
        # if write_log:
        #     log_path = os.path.join(data[section]["TokenizedDir"], section + "_sampled.json")
        #     with open(log_path, "w") as log_file:
        #         log_file.write(json.dumps(out[section], indent = 2))
    return out

def pick(dataset_name: str,
          write_log: bool = True,
          n: int = N,
          data: configparser.ConfigParser = DATA,
          y_var: str = Y_VAR) -> callable:
  cache = pick_from_each_dataset(dataset_name, 
                                  write_log=write_log,
                                  n=n,
                                  data=data,
                                  y_var=y_var)
  def get(t: str) -> list:
    nonlocal cache
    out = dict()
    for dataset in cache.keys():
      for train_test in cache[dataset].keys():
        if t in cache[dataset][train_test]:
          out[dataset + '_' + train_test] = cache[dataset][train_test][t]
    return out
  return get

def write(out_directory, label, picked):
  if not os.path.exists(out_directory):
    os.makedirs(out_directory)

  with open(out_directory + '_' + label.replace('/', '') + '.tsv', 'a') as out:
    first = True
    for key, samples in picked.items():
      if not first:
        out.write('{:-^500}'.format(''))
      first = False      
      for samp_dict in samples:
        out_str = '{:<40}\t'.format(key)
        out_str += '{:100}\t{:300}\t|{:300}\n'.format(samp_dict['mention'], 
                                                  ' '.join(samp_dict['left']), 
                                                  ' '.join(samp_dict['right']))
        out.write(out_str)

if __name__ == '__main__':
  dataset = 'BBN'
  label = '/PERSON'

  picker = pick(dataset, n=10)

  picked = picker(label)
  
  out_directory = BASE_OUT_PATH + dataset

  # write(out_directory, label, picked)