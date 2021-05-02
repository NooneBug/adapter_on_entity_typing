#!/usr/bin/env python3

from adapter_entity_typing.network_classes.classifiers import EarlyStoppingWithColdStart
from torch.utils.data.dataloader import DataLoader
from adapter_entity_typing.network import load_model, DEVICE
from collections import defaultdict
import torch
import json
import numpy as np
from tqdm import tqdm

import sys


sig = torch.nn.Sigmoid()
BATCH_SIZE = 100
WORKERS    = 20
stats_name = {
    "p": "precision",
    "r": "recall",
    "f1": "f1" }


def trimmed_stats(x, sampled = True):
    x_sorted = np.sort(x)[1:-1]
    return x_sorted.mean(), x_sorted.std(ddof = 1 if sampled else 0)


def compute_f1(p, r):
    return 2 * (p * r) / (p + r) if p + r else 0


def test(experiment):
    get_loader = lambda x: DataLoader(x, batch_size = BATCH_SIZE, num_workers = WORKERS)
    micros = {
        "p":  {"dev": [], "test": []},
        "r":  {"dev": [], "test": []},
        "f1": {"dev": [], "test": []}}
    macros = {
        "p":  {"dev": [], "test": []},
        "r":  {"dev": [], "test": []},
        "f1": {"dev": [], "test": []}}
    macro_examples = {
        "p":  {"dev": [], "test": []},
        "r":  {"dev": [], "test": []},
        "f1": {"dev": [], "test": []}}

    for model, _, dev_dataset, test_dataset, label2id in load_model(experiment):
        id2label = {v: k for k,v in label2id.items()}
        data_to_pred = [get_loader(test_dataset)]
        if model.configuration("DevOrTest") == "both":
            data_to_pred.append(get_loader(dev_dataset))

        for d, loader in zip(["test", "dev"], data_to_pred):
            all_preds = []
            all_preds_and_logits = []
            all_labels = []
            top_k_labels = []

            for mention, attn, labels in loader:
                mention = mention.to(DEVICE)
                attn    = attn.to(DEVICE)
                preds   = preds.to(DEVICE)

                batch_preds = []
                batch_preds_and_logits = []
                batch_top_k_labels = []

                for i, pred in enumerate(preds):
                    mask = pred > .5
                    ex_preds = []
                    ex_preds_and_logits = []
                    preds_ids = mask.nonzero()
                    no_pred = True
                    for p in pred_ids:
                        ex_preds.append(id2label[p.item()])
                        ex_preds_and_logits.append((id2label[p.item()],
                                                    round(preds[i][p].item(), 3)))
                        no_preds = False
                    topk_values, topk_indexes = torch.topk(pred, k = 5)
                    top_k_l = []
                    for val, index in zip(topk_values, topk_indexes):
                        val = round(val.item(), 3)
                        lab = id2label[index.item()]
                        top_k_l.append((lab, val))

                    if no_pred:
                        ex_preds.append(top_k_l[0][0])
                        ex_preds_and_logits.append(top_k_l[0])
                    sorted_ex_preds_and_logits = sorted(ex_preds_and_logits,
                                                        key=lambda x: x[1],
                                                        reverse=True)
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
            actual_count  = defaultdict(int)
            predict_count = defaultdict(int)

            bar = tqdm(desc="Computing macro performances", total=len(all_preds))
            for labels, preds in zip(all_labels, all_preds):
                for pred in preds:
                    predict_count[pred] += 1
                    if pred in labels:
                        correct_count[pred] += 1
                for label in labels:
                    actual_count[label] += 1
                bar.update(1)
            bar.close()

            precisions = {k: correct_count[k] / predict_count[k]
                          if predict_count[k] else 0
                          for k in label2id.keys()}
            recalls = {k: correct_count[k] / actual_count[k]
                       if actual_count[k] else 0
                       for k in label2id.keys()}
            f1s = {k: compute_f1(precisions[k], recalls[k])
                   for k in label2id.keys()}
            
            macro_p = np.mean(list(precisions.values()))
            macro_r = np.mean(list(recalls.values()))
            macro_f1 = compute_f1(macro_p, macro_r)
            
            macros['p'][d].append(macro_p)
            macros['r'][d].append(macro_r)
            macros['f1'][d].append(macro_f1)
            
            #compute macro_example performances
            ma_e_precisions = []
            ma_e_recalls = []
            n = len(all_labels)
            
            bar = tqdm(desc="computing macro examples performances", total=len(all_preds))
            for labels, preds in zip(all_labels, all_preds):
                correct_preds = len(set(labels).intersection(set(preds)))
                ma_e_precisions.append(correct_preds / len(preds))
                ma_e_recalls.append(correct_preds / len(labels))
                bar.update(1)
            bar.close()

            macro_example_p  = np.mean(ma_e_precisions)
            macro_example_r  = np.mean(ma_e_recalls)
            macro_example_f1 = compute_f1(macro_example_p, macro_example_r)

            macro_examples["p"][d].append(macro_example_p)
            macro_examples["r"][d].append(macro_example_r)
            macro_examples["f1"][d].append(macro_example_f1)

            bar = tqdm(desc="computing micro performances", total=len(all_preds))       
            for labels, preds in zip(all_labels, all_preds):
                micro_true_counter += len(labels)
                micro_pred_counter += len(preds)
                correct_preds = len(set(labels).intersection(set(preds)))
                micro_correct_counter += correct_preds
                bar.update(1)
            bar.close()
            micro_p = micro_correct_counter/micro_pred_counter
            micro_r = micro_correct_counter/micro_true_counter
            micro_f1 = compute_f1(micro_p, micro_r)
        
            micros['p'][d].append(micro_p)
            micros['r'][d].append(micro_r)
            micros['f1'][d].append(micro_f1)


            
            with open(dataset_paths[dataset_id], 'r') as inp:
                lines = [json.loads(l) for l in inp.readlines()]
            
            label_sentences = defaultdict(list)
            label_sentences = defaultdict(list)

            bar = tqdm(desc="generating sentences", total=len(lines))
            for l, preds_and_logits, top_k in zip(lines, all_preds_and_logits, top_k_labels):
                sentence = ' '.join(l['left_context_token'])
                sentence += ' ' + l['mention_span'] + ' '
                sentence += ' '.join(l['right_context_token'])
                labels = l['y_str']

                for lab in labels:
                    label_sentences[lab].append((sentence, l['mention_span'], preds_and_logits, top_k, labels))
                bar.update(1)
            bar.close()

            ordered_labels = list(sorted(label2id.keys()))

            # TODO: scrivere log delle risultati
        # TODO: scrivere file con indici
    # TODO: scrivere statistiche
