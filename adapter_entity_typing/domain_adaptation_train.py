from torch.utils.data.dataloader import DataLoader
from adapter_entity_typing.network import add_classifier
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from adapter_entity_typing.network_classes.classifiers import FromScratchPLWrapper, L2AWE, INCR_AD, FTAD
from adapter_entity_typing.train import declare_callbacks_and_trainer, get_random_seed
from adapter_entity_typing.domain_adaptation_utils import load_checkpoint, get_simple_model, parameter_type, AdapterType, ADAPTER_CONFIGS 
from adapter_entity_typing.utils import prepare_entity_typing_dataset_and_sample, get_label2id

if torch.cuda.is_available():
    print('gpu on')


def train_from_scratch(train_name, test_name, mode, configuration, conf_dict):
    overwrited_native = False
    print("Starting " + test_name + "\n\n")
    for i, s in enumerate(get_random_seed(), 1):
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        np.random.seed(s)
        random.seed(s)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        
        if overwrited_native:
          conf_dict['train']['PathInputTrain'] = native_train
          conf_dict['train']['PathInputDev'] = native_dev
          conf_dict['train']['PathInputTest'] = native_test

        if mode == 'None' or mode == 'FTAD':  
          model = get_simple_model(train_name, test_name, configuration)
          if mode == 'None':
            try:
                pretrained_name = model.configuration("Traineds", "train")[i - 1]
            except IndexError:
                break
          else:
            try:
                pretrained_name = model.configuration("Traineds", "test")[i - 1]
            except IndexError:
                break
            model = load_checkpoint(train_name, test_name, i - 1, configuration, 
                                    classification_model = model)
          if mode == 'None':
            print("\n\nTraining {} for the {} time".format(train_name, i))
          else:
            print("\n\nTraining {} for the {} time".format(test_name, i))

          if os.path.isfile(pretrained_name):
              print("Skipping")
              continue
          
        elif mode == 'L2AWE' or mode == 'INCR_AD':
          model = load_checkpoint(train_name, test_name, i - 1, configuration)
          try:
              pretrained_name = model.configuration("Traineds", "test")[i - 1]
          except IndexError:
              break
          print("\n\nTraining {} for the {} time".format(test_name, i))
          if os.path.isfile(pretrained_name):
              print("Skipping")
              continue
        
        # load data and initialize classifier & dataloaders
        if not overwrited_native:
          overwrited_native = True
          native_train = conf_dict['train']['PathInputTrain']
          native_dev = conf_dict['train']['PathInputDev']
          native_test = conf_dict['train']['PathInputTest']

        conf_dict['train']['PathInputTrain'] = conf_dict['test']['PathInputTrain']
        conf_dict['train']['PathInputDev'] = conf_dict['test']['PathInputDev']
        conf_dict['train']['PathInputTest'] = conf_dict['test']['PathInputTest']
        conf_dict['train']['ExperimentName'] = test_name


        def get_parameter(p: str, train_or_test_get: str = 'train'):
          nonlocal conf_dict
          return parameter_type.get(p, str)(conf_dict[train_or_test_get][p])
        model.configuration = get_parameter
        train_dataset, dev_dataset, test_dataset, label2id = sample_train_and_dev(model)
        if mode == 'FTAD' or mode == 'INCR_AD':
          add_classifier(model = model.classifier, labels = label2id, overwrite_ok = True)
        elif mode == 'None':
          add_classifier(model = model, labels = label2id)

        train_dataset.label_number = len(label2id)

        batch_size = model.configuration('BatchSize', 'train')
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=20, pin_memory = True)
        dev_loader = DataLoader(dev_dataset, batch_size = 100, num_workers=20)
        test_loader = DataLoader(test_dataset, batch_size = 100, num_workers=20)

        multi_val_loader = [dev_loader, test_loader]

        # start training :D
        id2label = {v: k for k,v in label2id.items()}
        if mode == 'None':
          pl_wrapper = FromScratchPLWrapper(model, id2label)
        elif mode == 'L2AWE':
          pl_wrapper = L2AWE(model, id2label)
        elif mode == 'INCR_AD':
          pl_wrapper = INCR_AD(model, id2label)
        elif mode == 'FTAD':
          pl_wrapper = FTAD(model, id2label)
        
        trainer = declare_callbacks_and_trainer(model)
        trainer.fit(pl_wrapper, train_loader, multi_val_loader)
        print("Saving on " + pretrained_name)

        # if you have enough, stop it
        if i >= model.configuration("n", "train"):
            break

def sample_train_and_dev(model):
    label2id = get_label2id(model)
    train_dataset, dev_dataset, test_dataset = prepare_entity_typing_dataset_and_sample(model,
                                                                            "train", 
                                                                            label2id)
    return train_dataset, dev_dataset, test_dataset, label2id

if __name__ == "__main__":
    import sys
    train_from_scratch(sys.argv[1])

