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

from utils import prepare_entity_typing_dataset
from network_classes.classifiers import adapterPLWrapper


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
        "PathInputTest":       str,     # path of the input test dataset
        "PathOutput":          str,     # path of the output
        "PathModel":           str,     # path for storing the adapeters weights
        "PretrainedModel":     str,     # the experiment setup for the pretrained model
        "DatasetName":         str,     # the name of the dataset; empty to avoid storing
        "DatasetTokenizedDir": str,     # the directory where the tokenized dataset is (will be) stored
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
        "BertFineTuning":      int,     # how many Bert's transformer finetune (starting from the last)
        }
    #
    def get_parameter(p: str):
        nonlocal config, parameter_type
        return parameter_type.get(p, eval)(config[experiment][p])
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
    elif model.configuration('BertFineTuning'):
        # generate a partial string which will match with each parameter in the i-esim transformer
        layer_to_freeze = ['layer.{}.'.format(i) for i in range(0, 12 - model.configuration('BertFineTuning'))]
        layer_to_freeze.append('embeddings')
        for name, param in model.bert.named_parameters():
          if any(n in name for n in layer_to_freeze):
            param.requires_grad = False
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

    
def load_model(experiment_name: str,
               config_file: str = PARAMETERS,
               pretrained: str = "bert-base-uncased"):

    """Load the model for a given EXPERIMENT_NAME."""

    # initialize a casual model
    model = get_model(experiment_name, config_file, pretrained)
    pretrained_model = model.configuration("PretrainedModel")
    if pretrained_model == "same":
        pretrained_model = experiment_name
    
    # read training & development data
    train_dataset, dev_dataset, test_dataset, label2id = prepare_entity_typing_datasets(model)

    # add the classifier for the given data
    add_classifier(model, label2id)
    
    # load the .ckpt file with pre-trained weights (if exists)
    ckpt = os.path.join(model.configuration("PathModel"),
                        pretrained_model + ".ckpt")

    if os.path.isfile(ckpt):
        model = adapterPLWrapper.load_from_checkpoint(ckpt,
                                                      adapterClassifier = model,
                                                      id2label = {v: k for k, v in label2id.items()},
                                                      lr = model.configuration("LearningRate"))
    
    model.to(DEVICE)
    model.eval()
    
    return model, train_dataset, dev_dataset, test_dataset, label2id
