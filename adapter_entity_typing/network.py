#!/usr/bin/env python3

from torch._C import device
from transformers.modeling_bert import BertModelWithHeads
import configparser

import torch
from transformers import AdapterType
# https://github.com/Adapter-Hub/adapter-transformers/tree/master/notebooks
# https://github.com/Adapter-Hub/adapter-transformers/blob/master/notebooks/01_Adapter_Training.ipynb
from transformers import TrainingArguments
from transformers.adapter_config import PfeifferConfig, HoulsbyConfig


import os

# the parameters file
PARAMETERS = "parameters.ini"

# the device to use 
DEVICE = torch.device("cuda" if torch.cuda.is_available() \
                      else "cpu")

ADAPTER_CONFIGS = {
            "Pfeiffer": PfeifferConfig,
            "Houlsby" : HoulsbyConfig }

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
        "PathInputTrain":      str,     # path of the input training dataset
        "PathInputDev":        str,     # path of the input dev dataset
        "PathOutput":          str,     # path of the output
        "PathModel":           str,     # path for storing the adapeters weights
        "AdapterConfig":       str,     # configuration of the adapter (Pfeiffer or Houlsby)
        "ReductionFactor":     int,     # reduction factor of the adapter layer (768 / ReductionFactor = nodes)
        "ClassificatorLayers": int,     # number of layers in the classifier
        "Patience":            int,     # patience befor early stop
        "MaxEpoch":            int,     # (maximum) number of training epoches
        "BatchSize":           int,     # batch size for both training and inference
        "LearningRate":        float,   # learning rate
        "MaxContextSideSize":  int,     # max number of words in right and left context
        "MaxEntitySize":       int,     # max number of words in the entity mention (the last words will be cutted)
        "ExperimentName":      str,     # experiment name for tensorboard
        }
    #
    def get_parameter(p: str):
        nonlocal config, parameter_type
        return parameter_type.get(p, lambda x: x)(config[experiment][p])
    #
    return get_parameter


def get_model(experiment_name: str,
              config_file:str = PARAMETERS,
              pretrained:str = "bert-base-uncased"):
    """Build the model with the configuration for a given experiment."""
    #
    # https://docs.adapterhub.ml/classes/adapter_config.html#transformers.AdapterConfig
    # https://docs.adapterhub.ml/classes/model_mixins.html?highlight=add_adapter#transformers.ModelAdaptersMixin.add_adapter
    #
    model = BertModelWithHeads.from_pretrained(pretrained)
    model.experiment_name = experiment_name
    model.configuration = read_parameters(model.experiment_name,
                                          config_file)
    
    if model.configuration('ReductionFactor'):
        adapter_config = ADAPTER_CONFIGS[model.configuration("AdapterConfig")](
            reduction_factor = model.configuration("ReductionFactor"))
    
        model.add_adapter(experiment_name, AdapterType.text_task, adapter_config)
        model.train_adapter(experiment_name)
    else:
        model.freeze_model()
    return model

def add_classifier(model, labels: dict = {}):
    """Add a classifier to the given model and returns it"""
    
    model.add_classification_head(
        model.experiment_name,
        num_labels=len(labels),
        layers=model.configuration("ClassificatorLayers"),
        multilabel = True,
        id2label=labels)