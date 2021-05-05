#!/usr/bin/env python3

import configparser
import regex as re
from collections import defaultdict

import adapter_entity_typing.network
from train import train
from test  import test



if __name__ == "__main__":
    parameters = adapter_entity_typing.network.PARAMETERS

    tests = configparser.ConfigParser()
    tests.read(parameters["test"][0])
    experiments = tests.sections()[1:]
    
    experiments_per_dataset = defaultdict(list)
    for experiment in experiments:
        dataset = re.search(r"(?<=_)\w+", tests[experiment]["TrainingName"]).group()
        experiments_per_dataset[dataset].append(experiment)
        
    for experiment in \
        experiments_per_dataset["BBN"] + \
        experiments_per_dataset["Choi"]:
        # experiments_per_dataset["FIGER"] + \
        # experiments_per_dataset["OntoNotes"] + \
        
        configuration = adapter_entity_typing.network.read_parameters(
                                                                      experiment,
                                                                      "test",
                                                                      parameters
                                                                      )
        if not configuration("IsTrained?"):
            train(configuration("TrainingName"))
        test(experiment)
