from torch.nn.modules.loss import BCELoss
from torch.utils.data.dataloader import DataLoader
from adapter_entity_typing.utils import prepare_entity_typing_dataset
from adapter_entity_typing.network import get_model, add_classifier
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


if torch.cuda.is_available():
    print('gpu on')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')


def declare_callbacks_and_trainer(model):
    callbacks = []    
    experiment_name = model.configuration("ExperimentName")
    early_stopping_patience = model.configuration("Patience", "train")
    epochs = model.configuration("MaxEpochs", "train")
    cold_start = model.configuration("ColdStart", "train")
    early_stop_callback = EarlyStoppingWithColdStart(
                                        monitor='example_macro/macro_f1',
                                        min_delta=0.00,
                                        patience=early_stopping_patience,
                                        verbose=False,
                                        mode='max',
                                        strict=True,
                                        cold_start_epochs=cold_start)
    callbacks.append(early_stop_callback)
    checkpoint_callback = ModelCheckpoint(monitor='example_macro/macro_f1',
                                          dirpath=model.configuration("PathModel"),
                                          filename=experiment_name,
                                          mode='max',
                                          save_last=False)
    callbacks.append(checkpoint_callback)
    logger = TensorBoardLogger(model.configuration("LightningPath"),
                               name=experiment_name,
                               default_hp_metric=False)

    trainer = Trainer(callbacks=callbacks,
                      logger=logger,
                      gpus = 1, 
                      max_epochs=epochs,
                      # TODO mettere come parametri (?)
                      # TODO riguardare per Choi
                      limit_train_batches=300,
                      # limit_val_batches=.25,
                      precision = 16)
    return trainer



def get_pretrained_name(base_name, i):
    return "{}-v{}.ckpt".format(base_name, i) if i else "{}.ckpt".format(base_name)

  
def get_random_seed():
    i = 0
    while True:
        yield i if i else 236451  # https://www.wikidata.org/wiki/Q75179705
        i += 1


def train(experiment):
    for i, s in enumerate(get_random_seed(), 1):
        print("Training {} for the {} time".format(experiment, i))
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        np.random.seed(s)
        random.seed(s)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        
        print("Starting " + experiment)
        model = get_model(experiment)
        
        # if already trained
        if os.path.isfile(get_pretrained_name(model.configuration("PathPretrainedModel"), i)):
            print("Skipping")
            continue

        # load data and initialize classifier & dataloaders
        train_dataset, label2id = prepare_entity_typing_dataset(model, "train")
        dev_dataset,   label2id = prepare_entity_typing_dataset(model, "dev",   label2id)
        add_classifier(model = model, labels = label2id)

        train_dataset.label_number = len(label2id)

        batch_size = model.configuration('BatchSize')
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=20, pin_memory = True)
        dev_loader = DataLoader(dev_dataset, batch_size = batch_size, num_workers=20, shuffle = True, pin_memory = True)

        # start training :D
        id2label = {v: k for k,v in label2id.items()}
        pl_wrapper = adapterPLWrapper(model, id2label)
        
        trainer = declare_callbacks_and_trainer(model)
        trainer.fit(pl_wrapper, train_loader, dev_loader)

        # if you have enough, stop it
        if i >= model.configuration("n", "train"):
            break


if __name__ == "__main__":
    import sys
    train(sys.argv[1])
