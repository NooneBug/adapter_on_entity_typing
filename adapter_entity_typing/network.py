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
import regex as re
from adapter_entity_typing.utils import prepare_entity_typing_datasets, prepare_entity_typing_dataset_only_sentences_and_string_labels
from adapter_entity_typing.network_classes.classifiers import adapterPLWrapper, EarlyStoppingWithColdStart


# the parameters file
PARAMETERS = {
    "train": ("train.ini", True),
    "test":  ("test.ini",  True),
    "data":  ("data.ini",  True) }

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
    n = int(train_configuration["n"])
    in_folder = lambda x: os.path.join(folder, x)
    return [in_folder("{}-v{}.ckpt".format(pretrained_name, i) if i \
                      else "{}.ckpt".format(pretrained_name))
            for i in range(n)]



def read_parameters(experiment: str,
                    train_or_test: str,
                    configs: dict = PARAMETERS,
                    true_name: str = ""):
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
        if v[1]:
            config[k].read(v[0])
        else:
            config[k].read_string(v[0])
    #
    config[train_or_test][experiment]["ExperimentName"] = true_name if true_name \
        else experiment
    dataset_name = config[train_or_test][experiment]["DatasetName"]
    training_name = experiment if train_or_test == "train" \
        else config["test"][experiment]["TrainingName"]
    train = config["train"][training_name]
    train["PathInputTrain"] = config["data"][train["DatasetName"]]["Train"]
    train["PathInputDev"]   = config["data"][train["DatasetName"]]["Dev"]
    train["PathInputTest"]  = config["data"][train["DatasetName"]]["Test"]
    train["PathPretrainedModel"] = os.path.join(
        train["PathModel"],
        experiment)
    configuration = {"data":  config["data"][dataset_name],
                     "train": train}
    if train_or_test == "test":
        test = config["test"][experiment]
        test["PathInputTrain"] = config["data"][test["DatasetName"]]["Train"]
        test["PathInputDev"]   = config["data"][test["DatasetName"]]["Dev"]
        test["PathInputTest"]  = config["data"][test["DatasetName"]]["Test"]
        test["Traineds"] = repr(get_pretraineds(train, training_name))
        test["IsTrained?"] = repr(all([os.path.isfile(x) for x in test["Traineds"]]))
        configuration["test"] = test
    parameter_type = {
        # global
        "ExperimentName":      str,     # name of the experiment
        "DatasetName":         str,     # name of the dataset (train or test)
        #
        # train
        "PathModel":           str,     # path for storing the pretrained weights
        "PathPretrainedModel": str,     # the (base) pretrained name
        "LightningPath":       str,     # where to store tensorboard logs
        "PathInputTrain":      str,     # path of the train set
        "PathInputDev":        str,     # path of the dev set
        # 
        "MaxEntitySize":       int,     # max number of words in the entity mention (the last words will be cutted)
        "MaxEpoch":            int,     # (maximum) number of training epoches
        "n":                   int,     # number of istances for the model
        "Patience":            int,     # patience befor early stop
        "ColdStart":           int,     # coldstart for early stopping
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
        "DevOrTest":           str,     # where to test model (both or test)
        "Traineds":            eval,    # list of trained models
        "PathInputTest":       str,     # path of the test set
        "IsTrained?":          eval,    # all models are trained?
        #
        # data
        "Train":               str,     # path of the train set
        "Dev":                 str,     # path of the dev set
        "Test":                str,     # path of the test set
        "TokenizedDir":        str      # folder to save tokenized cache
        }
    #
    def get_parameter(p: str, train_or_test_get: str = train_or_test):
        nonlocal configuration, parameter_type
        return parameter_type.get(p, str)(configuration[train_or_test_get][p])
    #
    return get_parameter


def manipulate_config(config_file: str,
                      section: str,
                      new_name: str = "",
                      **others):
    """Manipulate a parameter file as you want"""
    config = configparser.ConfigParser()
    config.read(config_name)
    config_dict = dict(config[section])
    config_dict.update(others)
    out = ["[{}]".format(new_name if new_name else section)] + \
        ["{} = {}".format(k, v) in config_dict.items()]
    return ("\n".join(out), False)


def test_to_train_name(experiment_name):
    experiment_name = re.sub("FeatureExtraction", "_ft_0", experiment_name)
    experiment_name = re.sub(r"bertFT(\d+)",   r"bert_ft_\1", experiment_name)
    experiment_name = re.sub(r"adapters(\d+)", r"adapter_\1", experiment_name)
    experiment_name = re.sub("FIGER",     "figer", experiment_name)
    experiment_name = re.sub("Choi",      "choi",  experiment_name)
    experiment_name = re.sub("BBN",       "bbn",   experiment_name)
    experiment_name = re.sub("OntoNotes", "onto",  experiment_name)
    experiment_name = re.sub("trained_on_", "", experiment_name)
    experiment_name = re.sub(r"tested_on_.+$", "", experiment_name)
    return re.match(r"(?:adapter_\d+|bert_ft_\d+)_[^_]+", experiment_name).group(0)


def get_model(experiment_name: str,
              config_file: dict = PARAMETERS,
              pretrained: str = "bert-base-uncased"):
    """Build the model with the configuration for a given experiment."""
    #
    # https://docs.adapterhub.ml/classes/adapter_config.html#transformers.AdapterConfig
    # https://docs.adapterhub.ml/classes/model_mixins.html?highlight=add_adapter#transformers.ModelAdaptersMixin.add_adapter
    #
    new_experiment_name = test_to_train_name(experiment_name)
    config_file["train"] = manipulate_config(config_file["train"][0],
                                             experiment_name,
                                             new_experiment_name)
    model = BertModelWithHeads.from_pretrained(pretrained)
    model.experiment_name = new_experiment_name
    model.configuration = read_parameters(new_experiment_name,
                                          "train",
                                          config_file,
                                          experiment_name)
    
    if model.configuration("ReductionFactor"):
        adapter_config = ADAPTER_CONFIGS[model.configuration("AdapterConfig")](
            reduction_factor = model.configuration("ReductionFactor"))
    
        model.add_adapter(new_experiment_name, AdapterType.text_task, adapter_config)
        model.train_adapter(new_experiment_name)
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
    configuration = read_parameters(experiment_name, "test", config_file)
    training_name = test_to_train_name(configuration("TrainingName")) 
    config_test_str  = manipulate_config(config_file["test"][0],
                                         experiment_name,
                                         training_name,
                                         TrainingName = training_name)
    config_train_str = manipulate_config(config_file["train"][0],
                                         configuration("TrainingName"),
                                         training_name)
    #
    new_config_file = {
        "train": config_train_str,
        "test":  config_test_str,
        "data":  config_file["data"]}
    new_configuration = read_parameters(training_name,
                                        "test",
                                        new_config_file,
                                        experiment_name)
    #
    # initialize a casual model
    classification_model = get_model(training_name, new_config_file, pretrained)
    #
    # read training & development data
    train_dataset, dev_dataset, test_dataset, label2id = \
        prepare_entity_typing_datasets(classification_model)
    #
    # add the classifier for the given data
    add_classifier(classification_model, label2id)
    #
    # load the mapper if not native
    native_train    = configuration("DatasetName", "train")
    non_native_test = configuration("DatasetName", "test")
    non_native_dev  = native_train if configuration("DevOrTest") == "both" \
        else non_native_test
    if native_train != non_native_test:
        mapping = MAPPINGS[native_trai]()[non_native_test]
        # TODO chiarire
        dev_dataset  = prepare_entity_typing_dataset_only_sentences_and_string_labels(non_native_dev , classification_model)
        test_dataset = prepare_entity_typing_dataset_only_sentences_and_string_labels(non_native_test, classification_model)
    #
    # load the .ckpt file with pre-trained weights (if exists)
    for ckpt in configuration("Traineds"):
        print("Loading {}".format(ckpt))
        model = adapterPLWrapper.load_from_checkpoint(
            ckpt,
            adapterClassifier = classification_model,
            id2label = {v: k for k, v in label2id.items()})
        model.configuration = new_configuration
        model.to(DEVICE)
        model.eval()
        yield model, train_dataset, dev_dataset, test_dataset, label2id
