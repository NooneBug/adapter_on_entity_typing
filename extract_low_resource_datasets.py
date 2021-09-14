import json
from tqdm import tqdm
from collections import defaultdict
import copy

def load_dataset_and_extract_types(path):
    with open(path, 'r') as inp:
        lines = [json.loads(l) for l in tqdm(inp.readlines())]
    return lines

def count_types(list_of_types):
    all_types = [t for l in tqdm(list_of_types) for t in l['y_str']]
    type_counter = defaultdict(int)
    for t in tqdm(all_types):
        type_counter[t] += 1
    c = [(word, round(type_counter[word], 2)) for word in tqdm(set(all_types))]     
    return sorted(c, key = lambda x : x[1], reverse = False)

def extract_k(list_of_examples, k, types_number, p_extracted, p_extracted_dataset, p_extracted_lines):

    extracted = copy.deepcopy(p_extracted)
    extracted_dataset = copy.deepcopy(p_extracted_dataset)
    extracted_lines = copy.deepcopy(p_extracted_lines) 
    max_not_reached = {t:False for t in types_number.keys()}
    i = 0
    while not all(list(max_not_reached.values())):
        example = list_of_examples[i]
        types = example['y_str']
        line_index = example['line']
        extracted_types = [extracted[t] for t in types]
        if line_index not in p_extracted_lines:
            if any([e < k for e in extracted_types]):
                extracted_dataset.append(example)
                extracted_lines.append(line_index)
                for t in types:
                    extracted[t] += 1

                max_not_reached = {t:extracted[t] >= types_number[t] or extracted[t] >= k for t in types_number.keys()} 
        i += 1
    return extracted_dataset, extracted, extracted_lines


if __name__ == "__main__":
  # BBN

  train = '/datahdd/vmanuel/entity_typing_all_datasets/data/BBN/BBN/train_partitioned.json'
  train_dataset = load_dataset_and_extract_types(train)
  types_number = dict(count_types(train_dataset))

  dataset_10, extracted_10, lines_10 = extract_k(train_dataset, 10, types_number, defaultdict(int), [], [])
  dataset_20, extracted_20, lines_20 = extract_k(train_dataset, 20, types_number, extracted_10, dataset_10, lines_10)
  print('ciaone')