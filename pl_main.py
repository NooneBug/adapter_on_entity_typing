from torch.nn.modules.loss import BCELoss
from torch.utils.data.dataloader import DataLoader
from adapter_entity_typing.utils import prepare_entity_typing_dataset, save_dataset, get_discrete_pred, compute_metrics
from adapter_entity_typing.network import get_model, add_classifier
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import numpy as np
import random
from tqdm import tqdm
from adapter_entity_typing.network_classes.classifiers import adapterPLWrapper
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


if torch.cuda.is_available():
  print('gpu on')
  # torch.set_default_tensor_type('torch.cuda.FloatTensor')

def declare_callbacks_and_trainer(early_stopping_patience, epochs, experiment_name):
    callbacks = []

    early_stop_callback = EarlyStopping(
                                        monitor='example_macro/macro_f1',
                                        min_delta=0.00,
                                        patience=early_stopping_patience,
                                        verbose=False,
                                        mode='max'
                                        )
    callbacks.append(early_stop_callback)

    checkpoint_callback = ModelCheckpoint(monitor='example_macro/macro_f1',
                                          # dirpath='/datahdd/vmanuel/checkpoints/',
                                          dirpath='trained_models/',
                                          filename='model',
                                          mode='max',
                                          save_last=False)
    callbacks.append(checkpoint_callback)


    logger = TensorBoardLogger('lightning_logs', name=experiment_name, default_hp_metric=False)

    trainer = Trainer(callbacks=callbacks, logger = logger, gpus = 1, 
                      max_epochs=epochs)

    return trainer

exps = [i for i in range(0, 15 + 1)]
# exps = [15]
# for experiment in ["experiment_" + str(i) for i in exps]:
# for experiment in ["OnlyMention"]:
exps = [# "bert_ft_0_", "bert_ft_1_", "bert_ft_2_", "adapter_1_",
        "adapter_2_", "adapter_4_", "adapter_8_"
	# , "adapter_16_"
]
exps_datasets = [# "figer",
                 # "bbn",
                 # "onto",
                 "crowd"]

exps = [e + d for e in exps for d in exps_datasets]

for experiment in exps:

  torch.manual_seed(236451)
  torch.cuda.manual_seed(236451)
  np.random.seed(236451)
  random.seed(236451)
  torch.backends.cudnn.enabled=False
  torch.backends.cudnn.deterministic=True
  torch.cuda.empty_cache()
  
  print("Starting " + experiment)
  model = get_model(experiment)

  train_path = model.configuration('PathInputTrain')
  dev_path = model.configuration('PathInputDev')
  max_context_side_size = model.configuration('MaxContextSideSize')
  max_entity_size = model.configuration('MaxEntitySize')
  early_stopping_patience = model.configuration('Patience')
  epochs = model.configuration('MaxEpoch')
  exp_name = model.configuration('ExperimentName')
  lr = model.configuration('LearningRate')
  batch_size = model.configuration('BatchSize')

  # train_dataset, label2id = prepare_entity_typing_dataset(train_path, load=True)
  # dev_dataset, _ = prepare_entity_typing_dataset(dev_path, label2id = label2id, load=True)

  train_dataset, label2id = prepare_entity_typing_dataset(train_path, max_context_side_size = max_context_side_size, max_entity_size=max_entity_size)
  dev_dataset, _ = prepare_entity_typing_dataset(dev_path, label2id = label2id, max_context_side_size = max_context_side_size, max_entity_size=max_entity_size)

  # save_dataset(train_dataset, label2id, 'datasets/3_types_context1_train.pkl')
  # save_dataset(dev_dataset, label2id, 'datasets/3_types_context1_dev.pkl')  

  train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=20)
  dev_loader = DataLoader(dev_dataset, batch_size = batch_size, num_workers=20)

  id2label = {v: k for k,v in label2id.items()}

  add_classifier(model = model, labels = label2id)

  pl_wrapper = adapterPLWrapper(model, id2label, lr)

  trainer = declare_callbacks_and_trainer(early_stopping_patience=early_stopping_patience,
                                          epochs = epochs,
                                          experiment_name=exp_name)

  trainer.fit(pl_wrapper, train_loader, dev_loader)

  # train(train_loader, dev_loader, model, label2id)
