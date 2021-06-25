#!/usr/bin/env python3

import configparser
import regex as re
from collections import defaultdict

import adapter_entity_typing.network
from adapter_entity_typing.train import train
from adapter_entity_typing.test  import test
from adapter_entity_typing.fine_tuning_train import fine_tune as ft_train
from adapter_entity_typing.fine_tuning_test  import test  as ft_test

from adapter_entity_typing.network_classes.classifiers import adapterPLWrapper, EarlyStoppingWithColdStart
import os


def make_experiments(experiments: dict, dataset: list, train_fn, test_fn):
    # for dataset in datasets:
    for experiment in experiments[dataset]:
        configuration = adapter_entity_typing.network.read_parameters(experiment,
                                                                    "test",
                                                                    parameters)

        if not configuration("IsTrained?"):
            print("\n\ntraining {}\n\n".format(configuration("TrainingName")))
            train_fn(configuration("TrainingName"))
        print("\n\ntesting {}\n\n".format(experiment))
        test_fn(experiment)


if __name__ == "__main__":

    # GROUP 1
    parameters = adapter_entity_typing.network.PARAMETERS
    tests = configparser.ConfigParser()
    tests.read(parameters["test"][0])
    experiments = tests.sections()    
    experiments_per_dataset = defaultdict(list)
    
    experiments_to_do = []
    for experiment in experiments:
        train_dataset = experiment.split('trained_on')[1].split('_')[1]
        test_dataset = experiment.split('tested_on')[1].split('_')[1]
        try:
            filter_dataset = experiment.split('filtered_with')[1].split('_')[1]
        except:
            filter_dataset = None
        if (filter_dataset and train_dataset != test_dataset) or not filter_dataset:
            experiments_to_do.append(experiment)
    # 
    for experiment in experiments_to_do:
        dataset = re.search(r"(?<=_)\w+", tests[experiment]["TrainingName"]).group()
        experiments_per_dataset[dataset].append(experiment)
        
    for dataset in experiments_per_dataset.keys():
        make_experiments(experiments_per_dataset, dataset, train, test)


    # # TODO: GROUP 2
    # parameters_ft = adapter_entity_typing.network.FINE_TUNING_PARAMETERS
    # tests = configparser.ConfigParser()
    # tests.read(parameters_ft)
    # experiments = tests.sections()[1:]

    # experiments_per_dataset = defaultdict(list)
    # for experiment in experiments:
    #     dataset = experiment.split("_")[1]
    #     experiments_per_dataset[dataset].append(experiment.lower())

    # for dataset in experiments_per_dataset.keys():
    #     make_experiments(experiments_per_dataset, dataset, train_ft, test_ft)
