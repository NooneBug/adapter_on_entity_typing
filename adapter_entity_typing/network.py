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
PARAMETERS = {"train": "train.ini",
              "test":  "test.ini",
              "data":  "data.ini"}

# the device to use 
DEVICE = torch.device("cuda" if torch.cuda.is_available() \
                      else "cpu")

ADAPTER_CONFIGS = {
            "Pfeiffer": PfeifferConfig,
            "Houlsby" : HoulsbyConfig }

MAPPINGS = {
    "BBN":       import_bbn_mappings,
    "OntoNotes": import_ontonotes_mappings,
    "FIGER":     import_figer_mappings,
    "Choi":      import_choi_mappings }


def get_pretraineds(train_configuration, pretrained_name):
    folder = train_configuration["PathModel"]
    n = train_configuration["n"]
    in_folder = lambda x: os.path.join(folder, x)
    return [in_folder("{}-v{}.ckpt".format(pretrained_name, i) if i \
                      else "{}.ckpt".format(pretrained_name))
            for i in range(n)]



def read_parameters(experiment: str,
                    train_or_test: str,
                    configs: str = PARAMETERS):
    """Read the configuration for a given experiment.
    Example of use:
    ```
    params_experiment_1 = read_parameters("Experiment_1", "parameters.ini")
    params_experiment_1("LearningRate")  # => 1e-7 (float)
    ```"""
    if train_or_test not in ["train", "test"]:
        raise Exception("`train_or_test` must be either `train` or `test`")
    config = {k: configparser.ConfigParser()
              for k in configs.keys()}
    for k, v in configs.items():
        config[k].read(v)
    config[train_or_test][experiment]["ExperimentName"] = experiment
    dataset_name = config[train_or_test][experiment]["DatasetName"]
    data = config["data"][dataset_name]
    if train_or_test == "train":
        train = config["train"][experiment]
        config["train"][experiment]["PathInputTrain"] = data["Train"]
        config["train"][experiment]["PathInputDev"]   = data["Dev"]
        config["train"][experiment]["PathModel"] = os.path.join(
            train["PathModel"],
            experiment)
        configuration = {"train": config["train"][experiment],
                         "data":  data}
    else:
        test  = config["test"][experiment]
        training_name = test["TrainingName"]
        train = config["train"][training_name]
        config["test"][experiment]["PathInputTest"] = data["Test"]
        config["test"][experiment]["Traineds"] = get_pretraineds(train, training_name)
        config["test"]["IsTrained?"] = all(
            [os.path.isfile(x) for x in test["Traineds"]])
        configuration = {"train": config["train"][training_name],
                         "test":  config["test"][experiment],
                         "data":  data}
    parameter_type = {
        # global
        "ExperimentName":      str,     # name of the experiment
        "DatasetName":         str,     # name of the dataset (train or test)
        #
        # train
        "PathModel":           str,     # path for storing the pretrained weights
        "PathInputTrain":      str,     # path of the train set
        "PathInputDev":        str,     # path of the dev set
        "n":                   int,     # number of istances for the model
        # 
        "MaxEntitySize":       int,     # max number of words in the entity mention (the last words will be cutted)
        "MaxEpoch":            int,     # (maximum) number of training epoches
        "Patience":            int,     # patience befor early stop
        "BatchSize":           int,     # batch size for both training and inference
        "LearningRate":        float,   # learning rate
        # 
        "MaxContextSideSize":  int,     # max number of words in right and left context
        "ClassificatorLayers": int,     # number of layers in the classifier
        "BertFineTuning":      int,     # how many Bert's transformer finetune (starting from the last)
        "AdapterConfig":       str,     # configuration of the adapter (Pfeiffer or Houlsby)
        "ReductionFactor":     int,     # reduction factor of the adapter layer (768 / ReductionFactor = nodes)
        #
        # test
        "TrainingName":        str,     # name of the pretrained weights
        "Traineds":            list,    # list of trained models
        "PathInputTest":       str,     # path of the test set
        "IsTrained?":          bool,    # all models are trained?
        #
        # data
        "Train":               str,     # path of the train set
        "Dev":                 str,     # path of the dev set
        "Test":                str,     # path of the test set
        "TokenizedDir":        str,     # folder to save tokenized cache
        }
    #
    def get_parameter(p: str, train_or_test_get: str = train_or_test):
        nonlocal configuration, parameter_type
        return parameter_type.get(p, str)(configuration[train_or_test_get][p])
    #
    return get_parameter

def get_model(experiment_name: str,
              config_file: str = PARAMETERS,
              pretrained: str = "bert-base-uncased"):
    """Build the model with the configuration for a given experiment."""
    #
    # https://docs.adapterhub.ml/classes/adapter_config.html#transformers.AdapterConfig
    # https://docs.adapterhub.ml/classes/model_mixins.html?highlight=add_adapter#transformers.ModelAdaptersMixin.add_adapter
    #
    model = BertModelWithHeads.from_pretrained(pretrained)
    model.experiment_name = experiment_name
    model.configuration = read_parameters(model.experiment_name,
                                          "train",
                                          config_file)
    
    if model.configuration("ReductionFactor"):
        adapter_config = ADAPTER_CONFIGS[model.configuration("AdapterConfig")](
            reduction_factor = model.configuration("ReductionFactor"))
    
        model.add_adapter(experiment_name, AdapterType.text_task, adapter_config)
        model.train_adapter(experiment_name)
    elif model.configuration("BertFineTuning"):
        # generate a partial string which will match with each parameter in the i-esim transformer
        layer_to_freeze = ["layer.{}.".format(i) \
                           for i in range(0, 12 - model.configuration("BertFineTuning"))]
        layer_to_freeze.append("embeddings")
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
               config_file: dict = PARAMETERS,
               pretrained: str = "bert-base-uncased"):

    """Load the model for a given EXPERIMENT_NAME."""

    # initialize a casual model
    configuration = read_parameters(experiment_name, "test", config_file)
    classification_model = get_model(configuration("TrainingName"),
                                     config_file,
                                     pretrained)
    # read training & development data
    train_dataset, dev_dataset, test_dataset, label2id = \
        prepare_entity_typing_datasets(classification_model)

    # add the classifier for the given data
    add_classifier(classification_model, label2id)

    # load the mapper if not native
    native_train    = model.configuration("DatasetName", "train")
    non_native_test = model.configuration("DatasetName", "test")
    non_native_dev  = native_train if model.configuration("DevOrTest") == "both" else non_native_teste
    if native_train != non_native_test:
        mapping = MAPPINGS[native_trai]()[non_native_test]
        # TODO chiarire
        dev_dataset  = prepare_entity_typing_dataset_only_sentences_and_string_labels(non_native_dev , classification_model)
        test_dataset = prepare_entity_typing_dataset_only_sentences_and_string_labels(non_native_test, classification_model)
        
    # load the .ckpt file with pre-trained weights (if exists)
    for ckpt in configuration("Traineds"):
        print("Loading {}".format(ckpt))
        model = adapterPLWrapper.load_from_checkpoint(ckpt,
                                                      adapterClassifier = classification_model,
                                                      id2label = {v: k for k, v in label2id.items()},
                                                      lr = classification_model.configuration("LearningRate", "train"))
        model.configuration = configuration
            
        model.to(DEVICE)
        model.eval()
        yield model, train_dataset, dev_dataset, test_dataset, label2id
