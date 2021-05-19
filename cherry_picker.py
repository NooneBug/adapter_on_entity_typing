import json
import configparser
import regex as re
from numpy import random
from collections import *


DATA = "data.ini"
Y_VAR = "y_str"
N = 10
DATA = configparser.ConfigParser()
DATA.read(DATA)


def get_dataset(dataset_path: str) -> list:
    "Read a DATASET_PATH and return observations as a list."
    with open(dataset_path, "r") as dataset_file:
        dataset = [json.load(l) for l in dataset_file.read()]
    return dataset


def get_from_dataset(dataset: list) -> callable:
    "Get a sampler for a given DATASET."
    def get_class(c: str, n: int = 0) -> list:
        "Sample the DATASET obtaining N observation for the class C."
        dataset_class = [obs for obs in dataset if obs[Y_VAR] == c]
        if not n:
            return dataset_class
        n_i =  min(n, len(dataset_class))
        return list(random.choice(dataset_class, n_i, replace = False))
    return get_class


def get_classes_from_dataset(dataset: list, y_var: str = Y_VAR) -> Counter:
    "Get all classes (Y_VAR) from a given DATASET and their frequence."
    classes = reduce(lambda x, y: x + y, [obs[y_var] for obs in dataset])
    return Counter(classes)


def pick(t: str, dataset_name: str,
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
                      data.sections)
    out = dict()
    for section in sections:
        out[section] = dict()
        datasets = {"train": data[section]["Train"],
                    # "dev": data[section]["Dev"],
                    "test":  data[section]["Test"]}
        for train_dev_test, dataset_path in {tdt: p for p in datasets.items()
                                             if p != "None"}.items():
            dataset = get_dataset(dataset_path)
            getter = get_from_dataset(dataset)
            classes = get_classes_from_dataset(dataset, y_var)
            out[section][train_dev_test] = {c: getter(c, n)
                                            for c in classes.keys()}
    if write_log:
        log_path = os.path.join(data["TokenizedDir"], dataset_name + "_sampled.json")
        with open(log_path, "w") as log_file:
            log_file.append(json.dumps(out))
    return out
