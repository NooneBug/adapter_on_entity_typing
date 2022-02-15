import torch
from torch.nn import Sigmoid, Linear, Module
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout
import pytorch_lightning as pl
from pytorch_lightning.metrics import Metric
import numpy as np
from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time

# class SimpleClassifier(BertModelWithHeads):
#   def __init__(self, config):
#       super().__init__(config)
#       self.sig = Sigmoid()
    
#   def forward(self, input_ids, attention_mask):
#       out = super().forward(input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             return_dict=True)['logits']
#       return self.sig(out)

class EarlyStoppingWithColdStart(EarlyStopping):
    def __init__(self, monitor: str, min_delta: float, patience: int, verbose: bool, mode: str, strict: bool, cold_start_epochs: int = 0):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, strict=strict)
        self.cold_start_epoch_number = cold_start_epochs
    
    def on_validation_end(self, trainer, pl_module):
        if trainer.running_sanity_check:
            return
        elif pl_module.current_epoch < self.cold_start_epoch_number:
            return
        else:
            self._run_early_stopping_check(trainer, pl_module)

class adapterPLWrapper(pl.LightningModule):
  def __init__(self, adapterClassifier, id2label) -> None:
      super().__init__()

      self.classifier = adapterClassifier
      self.id2label = id2label

      self.lr = self.classifier.configuration("LearningRate", "train")
      self.criterion = BCEWithLogitsLoss(pos_weight=torch.full((len(id2label),), 1.))

      self.sig = Sigmoid()

      self.declare_metrics(self.id2label)

  def declare_metrics(self, id2label):
    self.my_metrics = MyMetrics(id2label=id2label)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer

  def forward(self, input_ids, attention_mask):
    out = self.classifier(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)
    return out['logits']
  
  def training_step(self, batch, batch_step):
    batched_sentences, batched_attn, batched_labels = batch

    model_output = self(batched_sentences, batched_attn)
    loss = self.criterion(model_output, batched_labels)

    self.log('losses/train_loss', loss.detach(), on_epoch=True, on_step=False)

    return loss
  
  def validation_step(self, batch, batch_step):
    batched_sentences, batched_attn, batched_labels = batch

    model_output = self(batched_sentences, batched_attn)
    val_loss = self.criterion(model_output, batched_labels)

    self.log('losses/val_loss', val_loss.detach(), on_epoch=True, on_step=False)

    sigmoided_output = self.sig(model_output.detach())
    self.update_metrics(pred = sigmoided_output, labels=batched_labels.detach())

    return val_loss
  
  def validation_epoch_end(self, out):
    self.compute_metrics()

  
  def compute_metrics(self):

    avg_pred_number, void_predictions, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, ma_p, ma_r, ma_f1, predicted_class_number = self.my_metrics.compute()

    self.log('micro/micro_f1', micro_f1)
    self.log('micro/micro_p', micro_p)
    self.log('micro/micro_r', micro_r)

    self.log('macro/macro_f1', macro_f1)
    self.log('macro/macro_p', macro_p)
    self.log('macro/macro_r', macro_r)

    self.log('example_macro/macro_f1', ma_f1)
    self.log('example_macro/macro_p', ma_p)
    self.log('example_macro/macro_r', ma_r)

    self.log('other_metrics/avg_pred_number', avg_pred_number)
    # self.log('other_metrics/void_predictions', void_predictions)
    self.log('other_metrics/predicted_class_number', predicted_class_number)

  def update_metrics(self, pred, labels):

    pred = self.get_discrete_pred(pred)
    labels = labels.int()
    
    self.my_metrics.update(preds=pred, target=labels)
  
  def get_discrete_pred(self, pred, threshold = 0.5):
    mask = pred > threshold

    ones = torch.ones(mask.shape).cuda()
    zeros = torch.zeros(mask.shape).cuda()

    discrete_pred = torch.where(mask, ones, zeros)

    max_values_and_indices = torch.max(pred, dim = 1)

    for dp, i in zip(discrete_pred, max_values_and_indices.indices):
        dp[i] = 1
    
    return discrete_pred

  def compute_f1(self, p, r):	
      return torch.true_divide((2*p*r), (p + r)) if (p + r) else 0


class FromScratchPLWrapper(adapterPLWrapper):
  def __init__(self, adapterClassifier, id2label) -> None:
      super().__init__(adapterClassifier, id2label)
      self.test_metrics = MyMetrics(id2label)
      self.initialize_external_log()
  
  def validation_step(self, batch, batch_step, dataloader_idx):
    if dataloader_idx == 0:
      super().validation_step(batch, batch_step)
    else:
      batched_sentences, batched_attn, batched_labels = batch

      model_output = self(batched_sentences, batched_attn)
      val_loss = self.criterion(model_output, batched_labels)

      self.log('losses/test_loss', val_loss.detach(), on_epoch=True, on_step=False)

      sigmoided_output = self.sig(model_output.detach())
      self.update_test_metrics(pred = sigmoided_output, labels=batched_labels.detach())

  def compute_test_metrics(self):

    avg_pred_number, void_predictions, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, ma_p, ma_r, ma_f1, predicted_class_number = self.test_metrics.compute()

    self.log('micro/test_micro_f1', micro_f1)
    self.log('micro/test_micro_p', micro_p)
    self.log('micro/test_micro_r', micro_r)

    self.log('macro/test_macro_f1', macro_f1)
    self.log('macro/test_macro_p', macro_p)
    self.log('macro/test_macro_r', macro_r)

    self.log('example_macro/test_macro_f1', ma_f1)
    self.log('example_macro/test_macro_p', ma_p)
    self.log('example_macro/test_macro_r', ma_r)

    self.log('other_metrics/test_avg_pred_number', avg_pred_number)
    # self.log('other_metrics/test_void_predictions', void_predictions)
    self.log('other_metrics/test_predicted_class_number', predicted_class_number)

    self.log_externally(avg_pred_number, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, ma_p, ma_r, ma_f1, predicted_class_number)

  def initialize_external_log(self):
    with open('domain_adaptation_results/training_logs/{}.txt'.format(self.classifier.configuration("ExperimentName", 'train') + '_' + str(self.classifier.configuration('SampleSize', 'test'))), 
              'a') as out:
      readable = time.ctime(time.time())
      out.write('\n{}\n'.format(readable))

  def log_externally(self, avg_pred_number, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, ma_p, ma_r, ma_f1, predicted_class_number):
    with open('domain_adaptation_results/training_logs/{}.txt'.format(self.classifier.configuration("ExperimentName", 'train') + '_' + str(self.classifier.configuration('SampleSize', 'test'))), 
              'a') as out:
      
      out_str = '{:.2f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n'.format(avg_pred_number, 
                                                                                                              micro_p, micro_r, micro_f1, 
                                                                                                              macro_p, macro_r, macro_f1, 
                                                                                                              ma_p, ma_r, ma_f1, 
                                                                                                              predicted_class_number)
      out.write(out_str)
  


  def update_test_metrics(self, pred, labels):

    pred = self.get_discrete_pred(pred)
    labels = labels.int()
    
    self.test_metrics.update(preds=pred, target=labels)
  
  def validation_epoch_end(self, out):
    self.compute_metrics()
    self.compute_test_metrics()

class HiddenLayer(Module):
  def __init__(self, in_dim, out_dim) -> None:
      super().__init__()
      self.hidden = Linear(in_dim, out_dim)
      self.dropout = Dropout(p = 0.1)
      self.dropout2 = Dropout(p = 0.1)
      self.act = ReLU()
  
  def forward(self, x):
    return self.dropout2(self.act(self.hidden(self.dropout(x))))

class L2AWE(FromScratchPLWrapper):
  def __init__(self, plWrapper, id2label) -> None:
      super().__init__(plWrapper.classifier, id2label)
      
      self.pretrained_label_number = len(plWrapper.id2label)

      for param in self.parameters():
        param.requires_grad = False
      
      self.hidden = HiddenLayer(768 + self.pretrained_label_number, 768)
      self.classification_layer = Linear(768, len(id2label))

  def forward(self, input_ids, attention_mask):
    out = self.classifier(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)

    _, hidden_layers_encodings = self.classifier.distilbert(input_ids = input_ids, 
                                                            attention_mask = attention_mask,
                                                            output_hidden_states = True)
    
    # extract the representation of the entities
    # the representations is the sum of the hidden representation of the last four layers
    # given the preprocessing in this project, the second token in input is always the first token of the entity
    # following Nozza et al 2021, the entity representation is the representation of its first token 
    
    # debora ha usato SOLO l'entità, valutare se estrarre i token e dare quelli in pasto a DistilBERT
    entity_representation = torch.sum(torch.stack([h[:, 1, :] for h in hidden_layers_encodings[-4:]]), dim=0)

    logits = out['logits']

    logits_and_entity_repr = torch.cat((entity_representation, logits), dim = 1)

    h = self.hidden(logits_and_entity_repr)
    
    return self.classification_layer(h)

class INCR_AD(FromScratchPLWrapper):
  def __init__(self, plWrapper, id2label) -> None:
      super().__init__(plWrapper.classifier, id2label)
      
      for name, param in self.named_parameters():
        if 'adapter' in name or 'heads' in name:
          param.requires_grad = True
        else:
          param.requires_grad = False

  def forward(self, input_ids, attention_mask):

    # extract the output from distilbert (with only the adapters unfreezed)
    # extract the CLS representation
    # use a HiddenLayer and a classification layer to classify
    
    out = self.classifier(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)
    return out['logits']

class FTAD(FromScratchPLWrapper):
  def __init__(self, plWrapper, id2label) -> None:
      super().__init__(plWrapper.classifier, id2label)
  
  def forward(self, input_ids, attention_mask):

    out = self.classifier(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)
    return out['logits']

class MyMetrics(Metric):
    def __init__(self, id2label ,dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.id2label=id2label
        self.label2id = {v:k for k,v in id2label.items()}

        self.pred_classes = []
        self.true_classes = []

        self.labels_pred_classes = []
        self.labels_true_classes = []


    def logits_and_one_hot_labels_to_string(self, logits, one_hot_labels, no_void = False, threshold = 0.5):	

        pred_classes, true_classes = [], []	

        for example_logits, example_labels in zip(logits, one_hot_labels):	
            mask = example_logits > threshold	
            if no_void:	
                argmax = np.argmax(example_logits)	
                pc = self.id2label[argmax]	
                p_classes = [self.id2label[i] for i, m in enumerate(mask) if m]	

                if pc not in p_classes:	
                    p_classes.append(pc)	
                pred_classes.append(p_classes)	

            else:
                true_indexes = mask.nonzero(as_tuple=True)[0]   	
                pred_classes.append([self.id2label[m.item()] for m in true_indexes])
            mask = example_labels > .5
            true_indexes = mask.nonzero(as_tuple=True)[0]
            true_classes.append([self.id2label[l.item()] for l in true_indexes])	

        assert len(pred_classes) == len(true_classes), "Error in id2label traduction"	
        return pred_classes, true_classes

    # def compute_metrics_old(self, pred_classes, true_classes):	
    #     correct_counter = 0	
    #     prediction_counter = 0	
    #     true_labels_counter = 0	
    #     precision_sum = 0	
    #     recall_sum = 0	
    #     f1_sum = 0	

    #     void_prediction_counter = 0	

    #     for example_pred, example_true in zip(pred_classes, true_classes):	

    #         assert len(example_true) > 0, 'Error in true label traduction'	

    #         prediction_counter += len(example_pred)	

    #         true_labels_counter += len(example_true)	
    #         if not example_pred:	
    #             void_prediction_counter += 1	
    #         else:	
    #             correct_predictions = len(set(example_pred).intersection(set(example_true)))	
    #             correct_counter += correct_predictions	

    #             p = correct_predictions / len(example_pred)	
    #             r = correct_predictions / len(example_true)	
    #             f1 = self.compute_f1(p, r)	
    #             precision_sum += p	
    #             recall_sum += r	
    #             f1_sum += f1

    #     if prediction_counter:
    #       micro_p = correct_counter / prediction_counter
    #     else:
    #       micro_p = 0
    #     micro_r = correct_counter / true_labels_counter	
    #     micro_f1 = self.compute_f1(micro_p, micro_r)	

    #     examples_in_dataset = len(true_classes)	

    #     macro_p = precision_sum / examples_in_dataset	
    #     macro_r = recall_sum / examples_in_dataset	
    #     macro_f1 = self.compute_f1(macro_p, macro_r)	

    #     avg_pred_number = prediction_counter / examples_in_dataset	


    #     return avg_pred_number, void_prediction_counter, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        p = self.infer_prediction(logits=preds)	
        self.pred_classes.extend(p)	
        self.true_classes.extend(target)

    def compute(self):
        assert len(self.pred_classes) == len(self.true_classes), "Error in id2label traduction"	

        avg_pred_number, void_predictions, p, r, f1, ma_t_p, ma_t_r, ma_t_f1, ma_e_p, ma_e_r, ma_e_f1 = self.compute_metrics(pred_classes=torch.stack(self.pred_classes),	
                                                                                                true_classes=torch.stack(self.true_classes).detach().cpu())
        # predicted_class_number = len(set([c for sub in self.pred_classes for c in sub ]))
        # predicted_class = [set(sub) for sub in self.pred_classes]
        # predicted_class_number = len(set.union(*predicted_class))

        predicted_class_number = torch.sum(torch.sum(torch.stack(self.pred_classes), dim = 0) > 0)
        
        self.pred_classes = []
        self.true_classes = []

        return avg_pred_number, void_predictions, p, r, f1, ma_t_p, ma_t_r, ma_t_f1, ma_e_p, ma_e_r, ma_e_f1, predicted_class_number

    def infer_prediction(self, logits, no_void = False, threshold = 0.5):	
        one_hot_preds = torch.zeros(size=logits.shape)
        masks = logits > threshold	
        for i, (mask, example_logits) in enumerate(zip(masks, logits)):	
            if no_void:	
                argmax = np.argmax(example_logits)	
                p_classes = mask.nonzero()

                if argmax not in p_classes:	
                    p_classes.append(argmax)	

            else:
                p_classes = mask.nonzero(as_tuple = True)[0] 	
            one_hot_preds[i][p_classes] = 1
        return one_hot_preds

    def zero_if_nan(self, tensor):
      return torch.where(tensor.isnan(), torch.zeros(size = tensor.shape), tensor)

    def compute_metrics(self, pred_classes, true_classes):

        one_h_sum = true_classes + pred_classes
        one_h_dif = true_classes - pred_classes

        # compute micro metrics
        correct = torch.sum(one_h_sum == 2)
        predicted = correct + torch.sum(one_h_dif == -1)
        groundtruth = correct + torch.sum(one_h_dif == 1)

        micro_precision = torch.true_divide(correct, predicted)
        micro_recall = torch.true_divide(correct, groundtruth)
        micro_f1 = self.compute_f1(micro_precision, micro_recall)

        # compute macro types metrics
        predicted_types = torch.sum(one_h_sum == 2, dim = 0) + torch.sum(one_h_dif == -1, dim = 0)
        correct_types = torch.sum(one_h_sum == 2, dim = 0)
        groundtruth_types = torch.sum(one_h_sum == 2, dim = 0) + torch.sum(one_h_dif == 1, dim = 0)

        macro_types_precision = torch.mean(self.zero_if_nan(torch.true_divide(correct_types, predicted_types)))
        macro_types_recall = torch.mean(self.zero_if_nan(torch.true_divide(correct_types, groundtruth_types)))
        macro_types_f1 = self.compute_f1(macro_types_precision, macro_types_recall)

        # compute macro example metrics
        predicted_examples = torch.sum(one_h_sum == 2, dim = 1) + torch.sum(one_h_dif == -1, dim = 1)
        correct_examples = torch.sum(one_h_sum == 2, dim = 1)
        groundtruth_examples = torch.sum(one_h_sum == 2, dim = 1) + torch.sum(one_h_dif == 1, dim = 1)

        macro_examples_precision = torch.mean(self.zero_if_nan(torch.true_divide(correct_examples, predicted_examples)))
        macro_examples_recall = torch.mean(self.zero_if_nan(torch.true_divide(correct_examples, groundtruth_examples)))
        macro_exmaples_f1 = self.compute_f1(macro_examples_precision, macro_examples_recall)

        prediction_number = torch.sum(pred_classes, dim = 1)
        avg_pred_number = torch.mean(prediction_number)
        void_prediction_counter = torch.sum(prediction_number == 0)

        return avg_pred_number, void_prediction_counter, \
            micro_precision, micro_recall, micro_f1, \
            macro_types_precision, macro_types_recall, macro_types_f1,\
            macro_examples_precision, macro_examples_recall, macro_exmaples_f1

    def compute_f1(self, p, r):	
        return torch.true_divide((2*p*r), (p + r)) if (p + r) else 0

