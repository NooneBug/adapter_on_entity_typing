#!/usr/bin/env python3

import configparser

import adapter_entity_typing
from adapter_entity_typing.domain_adaptation_train import train_from_scratch
from adapter_entity_typing.domain_adaptation_test  import test  as test_da
from adapter_entity_typing.domain_adaptation_utils import get_test_parameters, read_parameters

def make_experiments(experiments: dict, dataset: list, train_fn, test_fn, param):
    for experiment in experiments[dataset]:
        configuration = adapter_entity_typing.network.read_parameters(experiment,
                                                                    "test",
                                                                    param)

        if not configuration("IsTrained?"):
            print("\n\ntraining {}\n\n".format(configuration("TrainingName")))
            train_fn(configuration("TrainingName"))
        print("\n\ntesting {}\n\n".format(experiment))
        test_fn(experiment)

def make_from_scratch_experiments(experiments: dict, train_fn, test_fn, param):
    for experiment, mode in experiments:
        configuration, conf_dict = read_parameters(experiment,
                                        "test",
                                        mode,
                                        param)

        configuration, conf_dict  = get_test_parameters(experiment, "test", mode, conf_dict, param) 
        if not configuration("IsTrained?"):
            print("\n\ntraining {}\n\n".format(configuration("TrainingName")))
            train_fn(configuration("TrainingName"), experiment, mode, configuration, conf_dict)
        print("\n\ntesting {}\n\n".format(experiment))
        # test_fn(experiment)

# def make_domain_adaptation_experiments(experiments: dict, train_fn, test_fn, param):
#     for experiment in experiments:
#         configuration = read_parameters(experiment,
#                                         "test",
#                                         param)
#         if not configuration("IsTrained?"):
#             print("\n\ntraining {}\n\n".format(configuration("TrainingName")))
#             train_fn(configuration("TrainingName"), experiment)
#         print("\n\ntesting {}\n\n".format(experiment))
#         test_fn(experiment)

if __name__ == "__main__":

    # GROUP 1 (training from scratch)
    # parameters = adapter_entity_typing.network.PARAMETERS
    # tests = configparser.ConfigParser()
    # tests.read(parameters["test"][0])
    # experiments = tests.sections()    
    # experiments_per_dataset = defaultdict(list)
    
    # # 
    # for experiment in experiments:
    #     dataset = re.search(r"(?<=_)\w+", tests[experiment]["TrainingName"]).group()
    #     experiments_per_dataset[dataset].append(experiment)
        
    # for dataset in experiments_per_dataset.keys():
    #     make_experiments(experiments_per_dataset, dataset, train, test, parameters)


    # GROUP 2 (training from scratch on samples of datasets (equal to Group 1, but the references to the .ini files change))

    # parameters = adapter_entity_typing.domain_adaptation_utils.DOMAIN_ADAPTATION_PARAMETERS
    # tests = configparser.ConfigParser()
    # tests.read(parameters["test"][0])
    # experiments = tests.sections()

    # from_scratch_experiments = [(e, tests[e]['DomainAdaptationMode'])  for e in experiments if tests[e]['DomainAdaptationMode'] == 'None']

    # make_from_scratch_experiments(from_scratch_experiments, train_from_scratch, test_da, parameters)
    

    # # # GROUP 3 (domain adaptation approaches based on pretrained models)
    parameters = adapter_entity_typing.domain_adaptation_utils.DOMAIN_ADAPTATION_PARAMETERS
    tests = configparser.ConfigParser()
    tests.read(parameters["test"][0])
    experiments = tests.sections()

    experiments = tests.sections()

    domain_adaptation_experiments = [(e, tests[e]['DomainAdaptationMode']) for e in experiments if tests[e]['DomainAdaptationMode'] != 'None']

    make_from_scratch_experiments(domain_adaptation_experiments, train_from_scratch, test_da, parameters)