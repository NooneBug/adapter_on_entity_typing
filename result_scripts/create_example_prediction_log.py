from adapter_entity_typing.network_classes.classifiers import adapterPLWrapper
from torch.utils.data.dataloader import DataLoader
from adapter_entity_typing.utils import prepare_entity_typing_dataset
from adapter_entity_typing.network import get_model, add_classifier
import pickle
from collections import defaultdict
import torch
import json


## DEPRECATED
only_correct = True

model_path = 'trained_models/model-v1.ckpt'
classifier = get_model('OnlyMention')

train_path = classifier.configuration('PathInputTrain')
dev_path = classifier.configuration('PathInputDev')
max_context_side_size = classifier.configuration('MaxContextSideSize')
max_entity_size = classifier.configuration('MaxEntitySize')

train_dataset, label2id = prepare_entity_typing_dataset(train_path, max_context_side_size = max_context_side_size, max_entity_size=max_entity_size)
dev_dataset, label2id = prepare_entity_typing_dataset(dev_path, label2id = label2id, max_context_side_size = max_context_side_size, max_entity_size=max_entity_size)

dev_loader = DataLoader(dev_dataset, batch_size = 100, num_workers=20)

id2label = {v: k for k,v in label2id.items()}
vocab_len = len(id2label)

add_classifier(model = classifier, labels = label2id)

model = adapterPLWrapper.load_from_checkpoint(model_path, adapterClassifier = classifier, id2label = id2label, lr = 1e-4)
model.cuda()
model.eval()

train_dataset_stats_path = '../typing_network/typing_experiments/datasets_stats/bbn_train.pkl'
dev_dataset_stats_path = '../typing_network/typing_experiments/datasets_stats/bbn_dev.pkl'

sorted_cooc_dict_train = '../typing_network/datasets_stats/cooc_bbn_train.pkl'
sorted_cooc_dict_dev = '../typing_network/datasets_stats/cooc_bbn_dev.pkl'

metrics_file = 'result_scripts/results/onlyCorrect_sentences_onlyMention-BBN_prediction_log.txt'

dataset_path = '/datahdd/vmanuel/entity_typing_all_datasets/data/BBN/BBN/dev_partitioned.json'

all_preds = []
all_preds_and_logits = []
all_labels = []
top_k_labels = []
for mention, attn, labels in dev_loader:
    
    mention = mention.cuda()
    attn = attn.cuda()
    preds = model(mention, attn)
    
    batch_preds = []
    batch_preds_and_logits = []
    batch_top_k_labels = []
    for i, pred in enumerate(preds):
        mask = pred > .5
        ex_preds = []
        ex_preds_and_logits = []   
        pred_ids =  mask.nonzero()
        no_pred = True
        for p in pred_ids:
            ex_preds.append(id2label[p.item()])
            ex_preds_and_logits.append((id2label[p.item()], round(preds[i][p].item(), 3)))
            no_pred = False
        # sort logits by pred
        topk_values, topk_indexes = torch.topk(pred, k = 5)
        top_k_l = []
        for val, index in zip(topk_values, topk_indexes):
            val = round(val.item(), 3)
            lab = id2label[index.item()]
            top_k_l.append((lab, val))
        
        if no_pred:
          ex_preds_and_logits.append(top_k_l[0])

        sorted_ex_preds_and_logits = sorted(ex_preds_and_logits, key=lambda tup: tup[1], reverse = True)
        batch_preds.append(ex_preds)
        batch_preds_and_logits.append(sorted_ex_preds_and_logits)
        batch_top_k_labels.append(top_k_l)
    
    all_preds.extend(batch_preds)
    all_preds_and_logits.extend(batch_preds_and_logits)
    top_k_labels.extend(batch_top_k_labels)

    mask = labels == 1
    batch_labels = []
    for m in mask:
        ex_labels = []
        labels_ids = m.nonzero()
        for l in labels_ids:
            ex_labels.append(id2label[l.item()])
        batch_labels.append(ex_labels)
    all_labels.extend(batch_labels)

correct_count = defaultdict(int)
actual_count = defaultdict(int)
predict_count = defaultdict(int)

for labels, preds in zip(all_labels, all_preds):
    for pred in preds:
        predict_count[pred] += 1

        if pred in labels:
            correct_count[pred] += 1
    
    for label in labels:
        actual_count[label] += 1

def compute_f1(p, r):
    return (2*p*r)/(p + r) if p + r else 0

precisions = {k: correct_count[k]/predict_count[k] if predict_count[k] else 0 for k in label2id.keys()}
recalls = {k: correct_count[k]/actual_count[k] if actual_count[k] else 0 for k in label2id.keys()}
f1s = {k: compute_f1(precisions[k], recalls[k]) for k in label2id.keys()}

with open(dataset_path, 'r') as inp:
    lines = [json.loads(l) for l in inp.readlines()]


label_sentences = {k: [] for k in label2id.keys()}

for l, preds_and_logits, top_k in zip(lines, all_preds_and_logits, top_k_labels):
    sentence = ' '.join(l['left_context_token'])
    sentence += ' ' + l['mention_span'] + ' '
    sentence += ' '.join(l['right_context_token'])
    labels = l['y_str']

    for lab in labels:
        label_sentences[lab].append((sentence, l['mention_span'], preds_and_logits, top_k, labels))

ordered_labels = list(sorted(label2id.keys()))

with open(train_dataset_stats_path, 'rb') as inp:
    train_stats = pickle.load(inp)

with open(dev_dataset_stats_path, 'rb') as inp:
    dev_stats = pickle.load(inp)



with open(metrics_file, 'a') as out:
    out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('label_#', 'train_percent', 'val_percent', 'precision', 
                                                                    'recall', 'f1', 'sentence', 'mention', 
                                                                    'preds_and_logits', 'top_k_labels_and_logits', 'true_labels'))
    for label in ordered_labels:
        i = 0
        for sentence, mention, preds_and_logits, top_k, true_label in label_sentences[label]:
          if only_correct:
            pred_labels = [p for p, l in preds_and_logits]
            if label in pred_labels:
              out_string = '{}\t{:.4f}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{}\t{}\n'.format(label + '_' + str(i + 1),
                                                                                        train_stats[label],
                                                                                        dev_stats[label] if label in dev_stats else 0,
                                                                                        precisions[label],
                                                                                        recalls[label],
                                                                                        f1s[label],
                                                                                        sentence,
                                                                                        mention,
                                                                                        preds_and_logits,
                                                                                        top_k,
                                                                                        true_label)
              out.write(out_string)
              i += 1            
          else:
            out_string = '{}\t{:.4f}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{}\t{}\n'.format(label + '_' + str(i + 1),
                                                                                        train_stats[label],
                                                                                        dev_stats[label] if label in dev_stats else 0,
                                                                                        precisions[label],
                                                                                        recalls[label],
                                                                                        f1s[label],
                                                                                        sentence,
                                                                                        mention,
                                                                                        preds_and_logits,
                                                                                        top_k,
                                                                                        true_label)
            out.write(out_string)
            i += 1
