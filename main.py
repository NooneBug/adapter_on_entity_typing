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

if torch.cuda.is_available():
  print('gpu on')
  # torch.set_default_tensor_type('torch.cuda.FloatTensor')


torch.manual_seed(236451)
torch.cuda.manual_seed(236451)
np.random.seed(236451)
random.seed(236451)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

# #########
# add_classifier(model = model, labels = label2id)

# it = iter(train_loader)
# batch = next(it)
# batched_sentences, batched_attn, batched_labels = batch

# outputs = model(batched_sentences, attention_mask = batched_attn)

# #########

def train(train_loader, dev_loader, model, label2id):
  add_classifier(model = model, labels = label2id)
  id2label = {v: k for k, v in label2id.items()}
  
  if torch.cuda.is_available():
    model.to('cuda')

  max_epochs = model.configuration('MaxEpoch')
  lr = model.configuration('LearningRate')
  max_patience = model.configuration('Patience')

  epoch = 0
  early_stop = False
  criterion = BCELoss()
  optimizer = Adam(lr = lr, params=model.parameters())

  while epoch < max_epochs and not early_stop:
    
    model.train()
    running_loss = 0.0
    steps = 0

    bar = tqdm(total=len(train_loader), desc='Training')

    for batch in train_loader:
      batched_sentences, batched_attn, batched_labels = batch
      
      if torch.cuda.is_available():
        batched_sentences = batched_sentences.cuda()
        batched_attn = batched_attn.cuda()
        batched_labels = batched_labels.cuda()
    
      outputs = model(batched_sentences, attention_mask = batched_attn)

      optimizer.zero_grad()
      loss = criterion(outputs, batched_labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      steps += 1
      bar.update(1)
    
    bar.close()
    
    loss = running_loss/steps

    with torch.no_grad():
      
      bar = tqdm(total=len(dev_loader), desc='Eval')

      model.eval()
      dev_running_loss = 0.0
      dev_steps = 0

      all_discrete_preds = []
      all_labels = [] 

      for batch in dev_loader:
        batched_sentences, batched_attn, batched_labels = batch

        if torch.cuda.is_available():
          batched_sentences = batched_sentences.cuda()
          batched_attn = batched_attn.cuda()
          batched_labels = batched_labels.cuda()

        outputs = model(batched_sentences, attention_mask = batched_attn)
        
        optimizer.zero_grad()
        dev_loss = criterion(outputs, batched_labels)
        dev_running_loss += dev_loss.item()

        dev_steps += 1
      
        discrete_pred = get_discrete_pred(outputs.detach().cpu().numpy(), id2label=id2label)
        discrete_lab = get_discrete_pred(batched_labels.detach().cpu().numpy(), id2label=id2label)

        all_discrete_preds.extend(discrete_pred)
        all_labels.extend(discrete_lab)

        bar.update(1)
      
    bar.close()
    dev_loss = dev_running_loss / dev_steps

    avg_pred_number, void_prediction_counter, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1 = compute_metrics(all_discrete_preds, 
                                                                                                                        all_labels)

    if epoch == 0:
      best_macro_f1 = macro_f1
      patience = 0
    else:
      if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        patience = 0
      else: 
        patience += 1
        if patience >= max_patience:
          early_stop = True

    print('epoch: {}; loss: {:.2f}; val_loss: {:.2f}; macro_f1: {:.2f}; best_macro_f1: {:.2f}, patience: {}'.format(epoch, 
                                                                                              loss, 
                                                                                              dev_loss, 
                                                                                              macro_f1,
                                                                                              best_macro_f1,
                                                                                              patience))
    
    if early_stop:
      print('early stopped')

    epoch += 1



for experiment in ["experiment_" + str(i) for i in range(0, 15 + 1)]:
  print("Starting " + experiment)
  model = get_model(experiment)

  with open("output.log", "a") as log_file:
    log_file.write("Experiment: {}\n".format(experiment))

  train_path = model.configuration('PathInputTrain')
  dev_path = model.configuration('PathInputDev')
  max_context_side_size = model.configuration('MaxContextSideSize')
  max_entity_size = model.configuration('MaxEntitySize')

  # train_dataset, label2id = prepare_entity_typing_dataset(train_path, load=True)
  # dev_dataset, _ = prepare_entity_typing_dataset(dev_path, label2id = label2id, load=True)

  train_dataset, label2id = prepare_entity_typing_dataset(train_path, max_context_side_size = max_context_side_size, max_entity_size=max_entity_size)
  dev_dataset, _ = prepare_entity_typing_dataset(dev_path, label2id = label2id, max_context_side_size = max_context_side_size, max_entity_size=max_entity_size)

  # save_dataset(train_dataset, label2id, 'datasets/3_types_context1_train.pkl')
  # save_dataset(dev_dataset, label2id, 'datasets/3_types_context1_dev.pkl')

  batch_size = model.configuration('BatchSize')

  train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
  dev_loader = DataLoader(dev_dataset, batch_size = batch_size)

  train(train_loader, dev_loader, model, label2id)
