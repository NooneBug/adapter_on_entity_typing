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
from result_scripts.import_mappings import import_bbn_mappings, import_choi_mappings, import_figer_mappings, import_ontonotes_mappings

import os

from adapter_entity_typing.utils import prepare_entity_typing_datasets, prepare_entity_typing_dataset_only_sentences_and_string_labels
from adapter_entity_typing.network_classes.classifiers import adapterPLWrapper, EarlyStoppingWithColdStart


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
        return parameter_type.get(p, str)(config[experiment][p])
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

def load_model_with_nonnative_datasets(experiment_name: str,
                                        config_file: str = PARAMETERS,
                                        training_file: str = PARAMETERS,
                                        pretrained: str = "bert-base-uncased"):

    """Load the model for a given EXPERIMENT_NAME."""

    # initialize a casual model
    test_configuration = read_parameters(experiment_name, config_file)
    classification_model = get_model(experiment_name, training_file, pretrained)
    classification_model.test_configuration = test_configuration
    pretrained_model = classification_model.configuration("PretrainedModel")
    configuration = classification_model.configuration
    if pretrained_model == "same":
        pretrained_model = classification_model.configuration("ExperimentName")
    pretrained_folder = os.path.dirname(classification_model.test_configuration("PathModel"))
    
    native_dataset_name = test_configuration("NativeDatasetName")
    if native_dataset_name == 'bbn':
        mappings = import_bbn_mappings()
    elif native_dataset_name == 'ontonotes':
        mappings = import_ontonotes_mappings()
    elif native_dataset_name == 'figer':
        mappings = import_figer_mappings()
    elif native_dataset_name == 'choi':
        mappings = import_choi_mappings()
    else:
        raise Exception('please provide a valid value for NativeDatasetName')
    
    nonnative_dataset_name = test_configuration("NonNativeDatasetName")
    if nonnative_dataset_name in ['bbn', 'figer', 'ontonotes', 'choi']:
        mapping_dict = mappings[nonnative_dataset_name]
    else:
        raise Exception('please provide a valid value for NonNativeDatasetName')

    nonnnative_dev = test_configuration("NonNativeDev")
    nonnnative_test = test_configuration("NonNativeTest")

    # read training & development data
    train_dataset, dev_dataset, test_dataset, label2id = prepare_entity_typing_datasets(classification_model)

    nonnative_dev_dataset = prepare_entity_typing_dataset_only_sentences_and_string_labels(nonnnative_dev, classification_model)
    nonnative_test_dataset = prepare_entity_typing_dataset_only_sentences_and_string_labels(nonnnative_test, classification_model, train_dev_test = 'test')

    # add the classifier for the given data
    add_classifier(classification_model, label2id)
    
    # load the .ckpt file with pre-trained weights (if exists)
    print(pretrained_model)
    ckpts = [os.path.join(pretrained_folder, x)
             for x in os.listdir(pretrained_folder)
             if x.startswith(pretrained_model)]
    print(ckpts)
    for ckpt in ckpts:
        model = adapterPLWrapper.load_from_checkpoint(ckpt,
                                                      adapterClassifier = classification_model,
                                                      id2label = {v: k for k, v in label2id.items()},
                                                      lr = classification_model.configuration("LearningRate"))
    
        model.to(DEVICE)
        model.eval()
        model.configuration = configuration
        yield model, nonnative_dev_dataset, nonnative_test_dataset, label2id, mapping_dict


def load_model(experiment_name: str,
               config_file: str = PARAMETERS,
               training_file: str = PARAMETERS,
               pretrained: str = "bert-base-uncased"):

    """Load the model for a given EXPERIMENT_NAME."""

    # initialize a casual model
    test_configuration = read_parameters(experiment_name, config_file)
    classification_model = get_model(experiment_name, training_file, pretrained)
    classification_model.test_configuration = test_configuration
    pretrained_model = classification_model.configuration("PretrainedModel")
    configuration = classification_model.configuration
    if pretrained_model == "same":
        pretrained_model = classification_model.configuration("ExperimentName")
    pretrained_folder = os.path.dirname(classification_model.test_configuration("PathModel"))
    
    # read training & development data
    train_dataset, dev_dataset, test_dataset, label2id = prepare_entity_typing_datasets(classification_model)

    # add the classifier for the given data
    add_classifier(classification_model, label2id)
    
    # load the .ckpt file with pre-trained weights (if exists)
    print(pretrained_model)
    ckpts = [os.path.join(pretrained_folder, x)
             for x in os.listdir(pretrained_folder)
             if x.startswith(pretrained_model)]
    print(ckpts)
    for ckpt in ckpts:
        model = adapterPLWrapper.load_from_checkpoint(ckpt,
                                                      adapterClassifier = classification_model,
                                                      id2label = {v: k for k, v in label2id.items()},
                                                      lr = classification_model.configuration("LearningRate"))
    
        model.to(DEVICE)
        model.eval()
        model.configuration = configuration
        yield model, train_dataset, dev_dataset, test_dataset, label2id
