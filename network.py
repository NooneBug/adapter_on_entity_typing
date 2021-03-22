#!/usr/bin/env python3

import configparser
import datasets

import os


PARAMETERS = "parameters.ini"


def read_parameters(experiment: str = "DEFAULT",
                    file_path: str = PARAMETERS):
    """Read the configuration for a given experiment.
    Example of use:
    ```
    params_experiment_1 = read_parameters("Experiment_1", "parameters.ini")
    params_experiment_1("LearningRate")  # => 1e-7 (float)
    ```"""
    config = configparser.ConfigParser()
    config.read(file_path)
    config[experiment]["PathModel"] = os.path.join(config[experiment]["PathModel"],
                                                   experiment)
    parameter_type = {
        "PathInput":    str,     # path of the input dataset
        "PathOutput":   str,     # path of the output
        "PathModel":    str,     # path for storing the adapeters weights
        "AdapterSize":  int,     # size of the adapter layer
        "Patience":     int,     # patience befor early stop
        "MaxEpoch":     int,     # (maximum) number of training epoches
        "LearningRate": float }  # learning rate
    

    def get_parameter(p: str):
        nonlocal config, parameter_type
        return parameter_type[p](config[experiment][p])

    return get_parameter


