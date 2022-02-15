import configparser
import os
from adapter_entity_typing.network import get_pretraineds, test_to_train_name, manipulate_config
from adapter_entity_typing.utils import prepare_entity_typing_datasets
from transformers.modeling_distilbert import DistilBertModelWithHeads
from adapter_entity_typing.network import ADAPTER_CONFIGS, add_classifier
from transformers import AdapterType
from adapter_entity_typing.network_classes.classifiers import adapterPLWrapper
import torch

BASE_INI_PATH = 'parameters/'
DOMAIN_ADAPTATION_PARAMETERS = {
                                "train": (BASE_INI_PATH + "domain_adaptation_train.ini", True),
                                "test":  (BASE_INI_PATH + "domain_adaptation_test_s_figer.ini",  True),
                                "data":  (BASE_INI_PATH + "data.ini",  True) 
                                }
DEVICE = torch.device("cuda" if torch.cuda.is_available() \
                      else "cpu")

def make_dir(x):
    if x[0] != '/':
        path = os.path.normpath(x).split(os.sep)
    else:
        path = os.path.normpath(x).split(os.sep)[1:]
    complete_path = [os.path.join(*path[0:i + 1]) for i in range(len(path))]
    for p in complete_path:
        if not os.path.isdir(p):
            os.mkdir(p)

def get_config(configs):
  # read the .ini files
  config = {k: configparser.ConfigParser()
            for k in configs.keys()}
  for k, v in configs.items():
      if v[1]:
          config[k].read(v[0])
      else:
          config[k].read_string(v[0])
  return config

parameter_type = {
    # global
    "ExperimentName":      str,     # name of the experiment, generated at runtime
    "DatasetName":         str,     # name of the dataset (train dataset or test dataset)
    #
    # train
    "PathModel":           str,     # path for the folder for storing the trained models
    "PathPretrainedModel": str,     # path for save the trained model
    "LightningPath":       str,     # where to store tensorboard logs
    "PathInputTrain":      str,     # path of the train set
    "PathInputDev":        str,     # path of the dev set
    # 
    "MaxEntitySize":       int,     # max number of words in the entity mention (the last words will be cutted)
    "MaxEpochs":           int,     # (maximum) number of training epoches
    "n_model_to_pretrain": int,     # number of instances to train on the source domain
    "Patience":            int,     # patience befor early stop
    "ColdStart":           int,     # coldstart for early stopping
    "LimitTrainBatches":   int,     # number of training set batches per epoch
    "LimitValBatches":     float,   # number of validation set batches per epoch
    "BatchSize":           int,     # batch size for both training and inference
    "LearningRate":        float,   # learning rate
    # 
    "MaxContextSideSize":  int,     # max number of words in right and left context
    "ClassificatorLayers": int,     # number of layers in the classifier
    "BertFineTuning":      int,     # how many Bert's transformer finetune (starting from the last)
    "AdapterConfig":       str,     # configuration of the adapter (Pfeiffer or Houlsby)
    "ReductionFactor":     int,     # reduction factor of the adapter layer (768 / ReductionFactor = nodes)
    "n":                   int,     # number of models to be trained
    #
    # test
    "TrainingName":        str,     # name of the training routine used to train the model which we want to test
    "DevOrTest":           str,     # where to test model (both or test)
    "Traineds":            eval,    # list of trained models
    "PathInputTest":       str,     # path of the test set
    "IsTrained?":          eval,    # all models are trained?
    "PredictionFile":      str,     # where to store y_hat
    "PerformanceFile":     str,     # where to store raw performance
    "AvgStdFile":          str,     # where to store (trimmed) mean and standard deviation of metrics
    #
    # data
    "Train":               str,     # path of the train set
    "Dev":                 str,     # path of the dev set
    "Test":                str,     # path of the test set
    "TokenizedDir":        str,     # folder to save tokenized cache
    #
    # adaptation
    "AdaptationsNumber":   int,     # number of adaptation training to perform
    "DomainAdaptationMode":str,     # the mode which drives the domain adaptation model instantiation
    "SampleSize":          str,     # size of the dataset sample
    }
def read_parameters(experiment: str,
                    train_or_test: str,
                    mode: str,
                    configs: dict = DOMAIN_ADAPTATION_PARAMETERS.copy(),
                    true_name: str = ""):
    """Read the configuration for a given experiment.
    Example of use:
    ```
    params_experiment_1 = read_parameters("Experiment_1", "parameters.ini")
    params_experiment_1("LearningRate")  # => 1e-7 (float)
    ```"""

    
    if train_or_test not in ["train", "test"]:
        raise Exception("`train_or_test` must be either `train` or `test`")
    
    config = get_config(configs)
    
    # create the directory 'x' if not exists


    config[train_or_test][experiment]["ExperimentName"] = true_name if true_name \
        else experiment
        
    # obtain the name of the training experiment
    training_name = experiment if train_or_test == "train" \
        else config["test"][experiment]["TrainingName"]
    
    #extract the parameters from the "training_name" TAG
    train = config["train"][training_name]
    test = config["test"][experiment]

    # generate the folder in which save the model to train and create if not exists
    train["PathPretrainedModel"] = os.path.join(
        train["PathModel"],
        experiment)
    make_dir(train["PathModel"])

    # obtain the paths to train/dev/test sets for the train procedure
    if train['DatasetName'] != 'None':
        train["PathInputTrain"] = config["data"][train["DatasetName"]]["Train"]
        train["PathInputDev"]   = config["data"][train["DatasetName"]]["Dev"]
        train["PathInputTest"]  = config["data"][train["DatasetName"]]["Test"]
    
    # read the filenames of the pretrained models into the folder "PathModel"
    if mode == 'None':
        train["Traineds"] = repr(get_pretraineds(train, 
                                                    training_name + '_{}'.format(test['SampleSize'])))
    else:
        train["Traineds"] = repr(get_pretraineds(train, 
                                                true_name or training_name))


    dataset_name = config[train_or_test][experiment]["DatasetName"]

    configuration = {"data":  config["data"][dataset_name],
                     "train": train}
    def get_parameter(p: str, train_or_test_get: str = train_or_test):
        nonlocal configuration
        return parameter_type.get(p, str)(configuration[train_or_test_get][p])
    #
    return get_parameter, configuration

def get_test_parameters(experiment,
                    train_or_test: str,
                    mode: str,
                    configuration: dict,
                    configs: dict = DOMAIN_ADAPTATION_PARAMETERS.copy(),
                    ):
    # call this method always after calling read_parameters(train_or_test = 'train') and pass configuration 
    train = configuration['train']

    config = get_config(configs)
    
    test = config["test"][experiment]

    training_name = test["TrainingName"]

    # obtain the paths to train/dev/test sets for the test procedure
    test["PathInputTrain"] = config["data"][test["DatasetName"]]["Train"]
    test["PathInputDev"]   = config["data"][test["DatasetName"]]["Dev"]
    test["PathInputTest"]  = config["data"][test["DatasetName"]]["Test"]
    
    # read the filenames of the pretrained models into the folder "PathModel"
    if mode == 'None':
      test["Traineds"] = repr(get_pretraineds(train, 
                                                    training_name))
    else:
      test["Traineds"] = repr(get_pretraineds(train, 
                                                    experiment + '_{}'.format(test['SampleSize'])))
    # check if all filename are saved (and so there not the necessity to train them)
    test["IsTrained?"] = repr(all(map(os.path.isfile, eval(test["Traineds"]))))
    
    make_dir(test["PerformanceFile"])
    make_dir(test["PredictionFile"])
    make_dir(test["AvgStdFile"])
    
    configuration["test"] = test

    def get_parameter(p: str, train_or_test_get: str = train_or_test):
        nonlocal configuration
        return parameter_type.get(p, str)(configuration[train_or_test_get][p])
    #
    return get_parameter, configuration

def load_checkpoint(train_name: str,
                    test_name : str,
                    model_number: int,
                    configuration, 
                      classification_model = None):
    
    # load a pretrained model on a dataset X, setup it to be adapted on dataset Y, 
    # setup the sampler on Y, return the model, the sampler, the devset of Y and the label2id dict

    """get_model, but it belives that it is native even when it is not"""
    # config_file = domain_adaptation_config.copy()
    # domain_adaptation = configparser.ConfigParser()
    # domain_adaptation.read(domain_adaptation_config['test'][0])
    # domain_adaptation = domain_adaptation[test_name]

    training_name_sigla = test_to_train_name(train_name)
    
    if not classification_model:
        classification_model = get_simple_model(train_name, test_name, configuration)
    
    # config_file["train"] = manipulate_config(config_file["train"][0],
    #                                          train_name,
    #                                          training_name_sigla,
    #                                          **dict(domain_adaptation))
    
    # _, conf_dict = read_parameters(training_name_sigla,
    #                                 train_or_test = "train",
    #                                 mode = 'None',
    #                                 configs = config_file,
    #                                 true_name = train_name)
    # configuration = get_test_parameters(test_name, 'test', 'None', conf_dict)

    # load best n checkpoints
    if 'bert_ft_' in training_name_sigla:
        model_name = '_'.join(training_name_sigla.split('_')[:3])
        dataset_name = training_name_sigla.split('_')[3]
    else:
        model_name = '_'.join(training_name_sigla.split('_')[:2])
        dataset_name = training_name_sigla.split('_')[2]
    losses_path = os.path.join(
        configuration('PerformanceFile', 'test'),
        "{}_trained_on_{}_tested_on_{}_test.txt".format(model_name, dataset_name, dataset_name))
    with open(losses_path, "r") as losses_file:
        # 2 = macro_example_f1; 5 = macro_f1; 8 = micro_f1
        losses = [float(x.split("\t")[5])
                  for x in losses_file.read().split("\n") if x]

    # load the best AdaptationsNumber checkpoints
    ckpts = list(zip(classification_model.configuration("Traineds", "train"), losses))
    ckpts.sort(key = lambda x: x[1], reverse = True)
    ckpts = [ckpt[0] for ckpt in ckpts[0:configuration("n", 'train')]]

    # load data

    _, _, _, label2id = prepare_entity_typing_datasets(classification_model)
    
    id2label = {v: k for k, v in label2id.items()}

    ckpt = ckpts[model_number]
    
    print("Loading {} as pretrained model # {}".format(ckpt, model_number + 1))

    
    add_classifier(model = classification_model, labels = label2id)

    model = adapterPLWrapper.load_from_checkpoint(
        ckpt,
        adapterClassifier = classification_model,
        id2label = id2label, 
        strict = False)
    
    model.to(DEVICE)
    model.configuration = configuration
    return model

def modify_train_config_with_domain_adaptation_params(train_name: str,
                                                      test_name: str,
                                                      config_file: dict = DOMAIN_ADAPTATION_PARAMETERS):
  config_file = config_file.copy()

  _, conf_dict = read_parameters(test_name, "test", 'None', config_file, train_name)
  test_config, _ = get_test_parameters(test_name, "test", 'L2AWE', conf_dict, config_file)
  
  new_experiment_name = test_to_train_name(train_name)
  if train_name != new_experiment_name or True:
      config_file["train"] = manipulate_config(config_file["train"][0],
                                                train_name,
                                                new_experiment_name,
                                                SampleSize = test_config("SampleSize"))
  return config_file, new_experiment_name

def get_simple_model(train_name: str,
              test_name: str,
              configuration,
              config_file: dict = DOMAIN_ADAPTATION_PARAMETERS,
              pretrained: str = "distilbert-base-uncased"):
              # pretrained: str = "bert-base-uncased"):
    """Build the model with the configuration for a given experiment."""
    #
    # https://docs.adapterhub.ml/classes/adapter_config.html#transformers.AdapterConfig
    # https://docs.adapterhub.ml/classes/model_mixins.html?highlight=add_adapter#transformers.ModelAdaptersMixin.add_adapter
    #
    # config_file, new_experiment_name = modify_train_config_with_domain_adaptation_params(train_name, test_name, config_file)

    model = DistilBertModelWithHeads.from_pretrained(pretrained)
    model.experiment_name = test_name
    # model.configuration, _ = read_parameters(new_experiment_name,
    #                                       "train",
    #                                       'None',
    #                                       config_file,
    #                                       train_name)
    model.configuration = configuration
    
    adapter_config = ADAPTER_CONFIGS[model.configuration("AdapterConfig", "train")](
        reduction_factor = 16)

    model.add_adapter(test_name, AdapterType.text_task, adapter_config)
    model.train_adapter(test_name)
    return model