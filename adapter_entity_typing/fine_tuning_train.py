from torch.nn.modules.loss import BCELoss
from torch.utils.data.dataloader import DataLoader
from adapter_entity_typing.utils import prepare_entity_typing_dataset
from adapter_entity_typing.network import get_model_to_finetune, add_classifier, PARAMETERS
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from adapter_entity_typing.network_classes.classifiers import adapterPLWrapper, EarlyStoppingWithColdStart
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from adapter_entity_typing.train import declare_callbacks_and_trainer, get_pretrained_name, get_random_seed

if torch.cuda.is_available():
    print('gpu on')


def fine_tune(experiment):
    print("Starting " + experiment + "\n\n")
    get_model = get_model_to_finetune(experiment) 
    for s in get_random_seed():
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        np.random.seed(s)
        random.seed(s)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()

        
        try:
            model, train_dataset, dev_dataset, label2id = next(get_model)
            print("\n\nTraining {} for the {} time".format(experiment, i))
        except StopIteration:
            break
            
        pretrained_name = model.configuration("Traineds", "train")[i - 1]
        if os.path.isfile(pretrained_name):
            print("Skipping")
            continue

        # load data and initialize classifier & dataloaders
        batch_size = model.configuration('BatchSize')
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=20, pin_memory = True)
        dev_loader = DataLoader(dev_dataset, batch_size = batch_size, num_workers=20, shuffle = True, pin_memory = True)

        # start training :D
        id2label = {v: k for k,v in label2id.items()}
        pl_wrapper = adapterPLWrapper(model, id2label)
        
        trainer = declare_callbacks_and_trainer(model)
        trainer.fit(pl_wrapper, train_loader, dev_loader)
        print("Saving on " + pretrained_name)


if __name__ == "__main__":
    import sys
    train(sys.argv[1])

