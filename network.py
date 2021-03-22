#!/usr/bin/env python3

import configparser
import datasets

import torch
import transformers
from transformers import BertModel, AdapterType
# https://github.com/Adapter-Hub/adapter-transformers/tree/master/notebooks
# https://github.com/Adapter-Hub/adapter-transformers/blob/master/notebooks/01_Adapter_Training.ipynb
from transformers import TrainingArguments, Trainer, EvalPrediction


import os

# the parameters file
PARAMETERS = "parameters.ini"

# the device to use 
DEVICE = torch.device("cuda" if torch.cuda.is_available() \
                      else "cpu")


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
        "NumLabels":    int,     # number of output labels
        "Patience":     int,     # patience befor early stop
        "MaxEpoch":     int,     # (maximum) number of training epoches
        "BatchSize":    int,     # batch size for both training and inference
        "LearningRate": float }  # learning rate
    #
    def get_parameter(p: str):
        nonlocal config, parameter_type
        return parameter_type[p](config[experiment][p])
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
    model = BertModel.from_pretrained(pretrained)
    model.experiment_name = experiment_name
    model.configuration = read_parameters(model.experiment_name,
                                          config_file)
    model.add_adapter(experiment_name, AdapterType.text_task)
    model.train_adapter(experiment_name)
    return model


def add_classifier(model, labels: dict = {}):
    """Add a classifier to the given model and returns it"""
    if labels:
        model.add_classification_head(
            model.experiment_name,
            num_labels=model.configuration("NumLabels"),
            id2label=labels)
    else:
        model.add_classification_head(
            model.experiment_name,
            num_labels=model.configuration("NumLabels"))



def train_model(model):
    training_args = TrainingArguments(
        learning_rate = model.configuration("LearningRate"),
        num_train_epochs = model.configuration("MaxEpoch"),
        per_device_train_batch_size=model.configuration("BatchSize"),
        per_device_eval_batch_size=model.configuration("BatchSize"),
        output_dir=model.configuration("PathOutput"))
    


bert = get_model("DEFAULT")
