import torch
from torch.nn import Sigmoid
from transformers.modeling_bert import BertModelWithHeads
import pytorch_lightning as pl
from pytorch_lightning.metrics import Metric
import numpy as np
from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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
    self.micro_precision = pl.metrics.classification.precision_recall.Precision(num_classes=len(self.id2label),
                                                                                    average='micro',
                                                                                    multilabel=True)
    # self.micro_precision = self.micro_precision.to('cpu')
    self.micro_recall = pl.metrics.classification.precision_recall.Recall(num_classes=len(self.id2label),
                                                                            average='micro',
                                                                            multilabel=True)
    # self.micro_recall = self.micro_recall.to('cpu')
    self.micro_f1 = pl.metrics.classification.F1(num_classes=len(self.id2label),
                                                    average='micro',
                                                    multilabel=True)
    # self.micro_f1 = self.micro_f1.to('cpu')
    self.macro_precision = pl.metrics.classification.precision_recall.Precision(num_classes=len(self.id2label),
                                                                                average='macro',
                                                                                multilabel=True)
    # self.macro_precision = self.macro_precision.to('cpu')
    self.macro_recall = pl.metrics.classification.precision_recall.Recall(num_classes=len(self.id2label),
                                                                                average='macro',
                                                                                multilabel=True)
    # self.macro_recall = self.macro_recall.to('cpu')
    self.macro_f1 = pl.metrics.classification.F1(num_classes=len(self.id2label),
                                                    average='macro',
                                                    multilabel=True)
    # self.macro_f1 = self.macro_f1.to('cpu')
    self.my_metrics = MyMetrics(id2label=id2label)
    # self.my_metrics = self.my_metrics.to('cpu')

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
    self.log('micro/micro_f1', self.micro_f1.compute())
    self.log('micro/micro_p', self.micro_precision.compute())
    self.log('micro/micro_r', self.micro_recall.compute())

    self.log('macro/macro_f1', self.macro_f1.compute())
    self.log('macro/macro_p', self.macro_precision.compute())
    self.log('macro/macro_r', self.macro_recall.compute())

    avg_pred_number, void_predictions, _, _, _, ma_p, ma_r, ma_f1, predicted_class_number = self.my_metrics.compute()

    self.log('example_macro/macro_f1', ma_f1)
    self.log('example_macro/macro_p', ma_p)
    self.log('example_macro/macro_r', ma_r)

    self.log('other_metrics/avg_pred_number', avg_pred_number)
    self.log('other_metrics/void_predictions', void_predictions)
    self.log('other_metrics/predicted_class_number', predicted_class_number)

  def update_metrics(self, pred, labels):

    pred = self.get_discrete_pred(pred)
    labels = labels.int()

    self.micro_f1.update(preds=pred, target=labels)
    self.micro_precision.update(preds=pred, target=labels)
    self.micro_recall.update(preds=pred, target=labels)

    self.macro_f1.update(preds=pred, target=labels)
    self.macro_precision.update(preds=pred, target=labels)
    self.macro_recall.update(preds=pred, target=labels)

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

class MyMetrics(Metric):
    def __init__(self, id2label ,dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.id2label=id2label
        self.label2id = {v:k for k,v in id2label.items()}

        self.pred_classes = []
        self.true_classes = []

        # self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.add_state("predicted", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        p, t = self.logits_and_one_hot_labels_to_string(logits=preds, one_hot_labels=target)	
        self.pred_classes.extend(p)	
        self.true_classes.extend(t)

    def compute(self):
        assert len(self.pred_classes) == len(self.true_classes), "Error in id2label traduction"	

        avg_pred_number, void_predictions, p, r, f1, ma_p, ma_r, ma_f1 = self.compute_metrics(pred_classes=self.pred_classes,	
                                                                                                true_classes=self.true_classes)
        # predicted_class_number = len(set([c for sub in self.pred_classes for c in sub ]))
        predicted_class = [set(sub) for sub in self.pred_classes]
        predicted_class_number = len(set.union(*predicted_class))
        
        self.pred_classes = []
        self.true_classes = []

        return avg_pred_number, void_predictions, p, r, f1, ma_p, ma_r, ma_f1, predicted_class_number

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

    def compute_metrics(self, pred_classes, true_classes):	
        correct_counter = 0	
        prediction_counter = 0	
        true_labels_counter = 0	
        precision_sum = 0	
        recall_sum = 0	
        f1_sum = 0	

        void_prediction_counter = 0	

        for example_pred, example_true in zip(pred_classes, true_classes):	

            assert len(example_true) > 0, 'Error in true label traduction'	

            prediction_counter += len(example_pred)	

            true_labels_counter += len(example_true)	
            if not example_pred:	
                void_prediction_counter += 1	
            else:	
                correct_predictions = len(set(example_pred).intersection(set(example_true)))	
                correct_counter += correct_predictions	

                p = correct_predictions / len(example_pred)	
                r = correct_predictions / len(example_true)	
                f1 = self.compute_f1(p, r)	
                precision_sum += p	
                recall_sum += r	
                f1_sum += f1

        if prediction_counter:
          micro_p = correct_counter / prediction_counter
        else:
          micro_p = 0
        micro_r = correct_counter / true_labels_counter	
        micro_f1 = self.compute_f1(micro_p, micro_r)	

        examples_in_dataset = len(true_classes)	

        macro_p = precision_sum / examples_in_dataset	
        macro_r = recall_sum / examples_in_dataset	
        macro_f1 = f1_sum / examples_in_dataset	

        avg_pred_number = prediction_counter / examples_in_dataset	


        return avg_pred_number, void_prediction_counter, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1

    def compute_f1(self, p, r):	
        return (2*p*r)/(p + r) if (p + r) else 0

