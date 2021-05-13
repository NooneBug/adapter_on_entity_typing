#!/usr/bin/env python3


from adapter_entity_typing.network_classes.classifiers import EarlyStoppingWithColdStart
from torch.utils.data.dataloader import DataLoader
from adapter_entity_typing.network import load_model_to_finetune, DEVICE
from collections import defaultdict
import torch
import json
import numpy as np
from tqdm import tqdm
import os

import sys

from adapter_entity_typing.test import sig, BATCH_SIZE, WORKERS, stats_name, trimmed_stats, compute_f1
from adapter_entity_typing.test import filter_label_and_return_translation
from adapter_entity_typing.test import filter_label_and_return_original
from adapter_entity_typing.test import take_first_k_filtered



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
    
    for model, dev_dataset, test_dataset, label2id, mapping in load_model_to_finetune(experiment):
        performance_file = os.path.join(
            model.configuration("PerformanceFile"),
            model.configuration("ExperimentName"))
        prediction_file = os.path.join(
            model.configuration("PredictionFile"),
            model.configuration("ExperimentName"))
        average_std_file = os.path.join(
            model.configuration("AvgStdFile"),
            model.configuration("ExperimentName"))
        id2label = {v: k for k,v in label2id.items()}
        
        data_to_pred = [get_loader(test_dataset)]
        if model.configuration("DevOrTest") == "both":
            data_to_pred.append(get_loader(dev_dataset))

        if mapping:
            if model.configuration("DatasetName", 'train') == model.configuration("DatasetName", 'test').split('_')[0]:
                filter_label = filter_label_and_return_original
            elif model.configuration("DatasetName", 'train') == model.configuration("DatasetName", 'test').split('_')[3]:
                filter_label = filter_label_and_return_translation

        for d, loader in zip(["test", "dev"], data_to_pred):
            all_preds = []
            all_preds_and_logits = []
            all_labels = []
            top_k_labels = []
            batch_start_index = 0
            
            bar = tqdm(total = len(loader.dataset.labels), desc = 'Generating predictions')

            for mention, attn, labels in loader:
                if mapping:
                    labels = loader.dataset.labels[batch_start_index: batch_start_index + BATCH_SIZE]
                    batch_start_index += BATCH_SIZE
                mention = mention.to(DEVICE)
                attn    = attn.to(DEVICE)
                preds = sig(model(mention, attn))
            
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
                    for p in preds_ids:
                        if mapping:
                            filtered_label = filter_label(id2label[p.item()], mapping)
                            ex_preds.extend(filtered_label)
                            if filtered_label:
                                ex_preds_and_logits.append((filtered_label, 
                                                            round(pred[p].item(), 3)))
                            no_pred = False
                        else:
                            ex_preds.append(id2label[p.item()])
                            ex_preds_and_logits.append((id2label[p.item()],
                                                        round(preds[i][p].item(), 3)))
                            no_pred = False
                    topk_values, topk_indexes = torch.topk(pred, k = 5) if not mapping \
                        else take_first_k_filtered(pred, 
                                                   mapping, 
                                                   id2label, 
                                                   k = 5,
                                                   filter_label = filter_label)
                    top_k_l = []
                    for val, index in zip(topk_values, topk_indexes):
                        val = round(val.item(), 3)
                        lab = id2label[index.item()] if not mapping \
                            else filter_label(id2label[index.item()], mapping)
                        if lab:
                          top_k_l.append((lab, val))

                    if no_pred:
                        ex_preds.append(top_k_l[0][0] if type(top_k_l[0][0]) == str else top_k_l[0][0][0])
                        ex_preds_and_logits.append(top_k_l[0] if type(top_k_l[0][0]) == str else (top_k_l[0][0][0], top_k_l[0][1]))
                    sorted_ex_preds_and_logits = sorted(ex_preds_and_logits,
                                                        key=lambda x: x[1],
                                                        reverse=True)
                    batch_preds.append(ex_preds)
                    batch_preds_and_logits.append(sorted_ex_preds_and_logits)
                    batch_top_k_labels.append(top_k_l)

                all_preds.extend(batch_preds)
                all_preds_and_logits.extend(batch_preds_and_logits)
                top_k_labels.extend(batch_top_k_labels)

                if mapping:
                  batch_labels = labels
                else:
                  mask = labels == 1
                  batch_labels = []
                  for m in mask:
                      ex_labels = []
                      labels_ids = m.nonzero()
                      for l in labels_ids:
                          ex_labels.append(id2label[l.item()])
                      batch_labels.append(ex_labels)
                all_labels.extend(batch_labels)
                bar.update(BATCH_SIZE)
            bar.close()
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

            if mapping:
              label_names = [l0 for l in label2id.keys() for l0 in filter_label(l, mapping_dict = mapping)]
            else:
              label_names = label2id.keys() 
            precisions = {k: correct_count[k] / predict_count[k]
                          if predict_count[k] else 0
                          for k in label_names}
            recalls = {k: correct_count[k] / actual_count[k]
                       if actual_count[k] else 0
                       for k in label_names}
            f1s = {k: compute_f1(precisions[k], recalls[k])
                   for k in label_names}
            
            macro_p = np.mean(list(precisions.values()))
            macro_r = np.mean(list(recalls.values()))
            macro_f1 = compute_f1(macro_p, macro_r)
            
            macros['p'][d].append(macro_p)
            macros['r'][d].append(macro_r)
            macros['f1'][d].append(macro_f1)
            
            #compute macro_example performances
            ma_e_precisions = []
            ma_e_recalls = []
            
            bar = tqdm(desc="computing macro examples performances", total=len(all_preds))
            for labels, preds in zip(all_labels, all_preds):
                correct_preds = len(set(labels).intersection(set(preds)))
                ma_e_precisions.append((correct_preds / len(preds)) if len(preds) else 0)
                ma_e_recalls.append(correct_preds / len(labels))
                bar.update(1)
            bar.close()

            macro_example_p  = np.mean(ma_e_precisions)
            macro_example_r  = np.mean(ma_e_recalls)
            macro_example_f1 = compute_f1(macro_example_p, macro_example_r)

            macro_examples["p"][d].append(macro_example_p)
            macro_examples["r"][d].append(macro_example_r)
            macro_examples["f1"][d].append(macro_example_f1)

            micro_true_counter = 0
            micro_pred_counter = 0
            micro_correct_counter = 0

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


            
            with open({"dev":  model.configuration("PathInputDev"),
                       "test": model.configuration("PathInputTest")}[d], "r") as inp:
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

            ordered_labels = list(sorted(label_names))
            with open(prediction_file + '_' + d + '.txt', 'a') as out:
                out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('label_#', 'precision', 
                                                                        'recall', 'f1', 'sentence', 'mention', 
                                                                        'preds_and_logits', 'top_k_labels_and_logits', 'true_labels'))
                for label in ordered_labels:
                    for i, (sentence, mention, preds_and_logits, top_k, true_label) in enumerate(label_sentences[label], 1):
                        out_string = '{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{}\t{}\n'.format(label + '_' + str(i),
                                                                                               precisions[label],
                                                                                               recalls[label],
                                                                                               f1s[label],
                                                                                               sentence,
                                                                                               mention,
                                                                                               preds_and_logits,
                                                                                               top_k,
                                                                                               true_label)
                        out.write(out_string)
            with open(performance_file + '_' + d + '.txt', 'a') as out:
                out.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(macro_example_p,
                                                                                                            macro_example_r,
                                                                                                            macro_example_f1,
                                                                                                            macro_p,
                                                                                                            macro_r,
                                                                                                            macro_f1,
                                                                                                            micro_p,
                                                                                                            micro_r,
                                                                                                            micro_f1))

    keys = ['test']
    if micros['f1']['dev']:
      keys = ['test', 'dev']

    for d in keys:
        results = {}
        for result_name, result in zip(["micro", "macro", "example"],
                                       [ micros,  macros, macro_examples]):
            print(result_name)
            print(result)
            print()
            for k, v in result.items():
                v = np.array(v[d])
                mu, sd = trimmed_stats(v)
                results["{}_{}".format(result_name, k)] = (mu, sd)

        with open(average_std_file + '_'+ d + '.txt', 'a') as out:
            out.write("model,mu,sd\n")
            for k, (m, s) in results.items():
                out.write('{},{:.4f},{:.4f}\n'.format(k, m, s))
            out.write('\n')



if __name__ == "__main__":
    import sys
    test(sys.argv[1])
