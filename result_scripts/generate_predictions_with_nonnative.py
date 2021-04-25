import configparser
from adapter_entity_typing.network_classes.classifiers import EarlyStoppingWithColdStart
from torch.utils.data.dataloader import DataLoader
from adapter_entity_typing.network import load_model, load_model_with_nonnative_datasets
from collections import defaultdict
import torch
import json
import numpy as np
from tqdm import tqdm


import sys


def filter_label(original_label, mapping_dict):
    if original_label in mapping_dict:
        return [original_label]
    else:
        return []


def take_first_k_filtered(predictions, mapping_dict, id2label, k):
    k_predicted_values = []
    k_predicted_idxs = []
    for predicted_value, value_id in zip(*torch.topk(predictions, k = len(predictions))):
        translation = filter_label(id2label[value_id.item()], mapping_dict)
        if translation:
            k_predicted_values.append(predicted_value.item())
            k_predicted_idxs.append(value_id.item())
        if len(k_predicted_values) >= k:
            break
    return k_predicted_values, k_predicted_idxs


parameter_tags = [sys.argv[1]]

config = configparser.ConfigParser()
config.read("result_scripts/generate_predictions_parameters.ini")
print(list(config.keys()))
config = config[parameter_tags[0]]

sig = torch.nn.Sigmoid()

micros = {
    "p": [],
    "r": [],
    "f1": []}
macros = {
    "p": [],
    "r": [],
    "f1": []}
macro_examples = {
    "p": [],
    "r": [],
    "f1": []}
    
experiment_name = config['experiment_name']
performance_file = config['performanceFile'] + parameter_tags[0]
prediction_file = config['predictionFile'] + parameter_tags[0]
average_std_file = config['AvgStdFile'] + parameter_tags[0]

dev_or_test = config['dev_or_test']
if dev_or_test == 'both':
    keys = ['dev', 'test']
elif dev_or_test == 'dev':
    keys = ['dev']
elif dev_or_test == 'test':
    keys = ['test']
else:
    raise Exception('please provide a meaningfull value for "dev_or_test"')

macros = {k: {subk: [] for subk in keys} for k, v in macros.items()}
micros = {k: {subk: [] for subk in keys} for k, v in macros.items()}
macro_examples= {k: {subk: [] for subk in keys} for k, v in macros.items()}

batch_size = 100

for model, dev_dataset, test_dataset, label2id, mapping_dict in load_model_with_nonnative_datasets(parameter_tags[0],
                                                                                                    experiment_name, 
                                                                                                    config_file = 'result_scripts/generate_predictions_parameters.ini'):  # , "results_scripts/generate_preditcions_parameters.ini"):

    dev_loader = DataLoader(dev_dataset, batch_size = batch_size, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=20)
    id2label = {v: k for k,v in label2id.items()}

    if dev_or_test == 'both':
        data_to_pred = ['dev', 'test']
        datasets = [dev_loader, test_loader]
        dataset_paths = [config['NonNativeDev'], config['NonNativeTest']]

    elif dev_or_test == 'dev':
        data_to_pred = ['dev']
        datasets = [dev_loader]
        dataset_paths = [config['NonNativeDev']]

    elif dev_or_test == 'test':
        data_to_pred = ['test']
        datasets = [test_loader]
        dataset_paths = [config['NonNativeTest']]

    else:
        raise Exception('please provide a meaningfull value for "dev_or_test"')

    for dataset_id, d in enumerate(data_to_pred):
        all_preds = []
        all_preds_and_logits = []
        all_labels = []
        top_k_labels = []
        loader = datasets[dataset_id]
        batch_start_index = 0

        for mention, attn in loader:
            batch_labels = loader.dataset.labels[batch_start_index: batch_start_index + batch_size]
            batch_start_index += batch_size
            mention = mention.cuda()
            attn = attn.cuda()
            preds = sig(model(mention, attn))
            
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
                    filtered_label = filter_label(id2label[p.item()], mapping_dict)
                    ex_preds.extend(filtered_label)
                    if filtered_label:
                        ex_preds_and_logits.append((filtered_label, 
                                                    round(pred[p].item(), 3)))
                    no_pred = False
                # sort logits by pred
                
                topk_values, topk_indexes = take_first_k_filtered(pred, 
                                                                    mapping_dict, 
                                                                    id2label, 
                                                                    k = 5)
                top_k_l = []
                for val, index in zip(topk_values, topk_indexes):
                    val = round(val, 3)
                    lab = filter_label(id2label[index], mapping_dict)
                    top_k_l.append((lab, val))
                
                if no_pred:
                    ex_preds.extend(top_k_l[0][0])
                    ex_preds_and_logits.append(top_k_l[0])

                sorted_ex_preds_and_logits = sorted(ex_preds_and_logits, key=lambda tup: tup[1], reverse = True)
                batch_preds.append(ex_preds)
                batch_preds_and_logits.append(sorted_ex_preds_and_logits)
                batch_top_k_labels.append(top_k_l)
            
            all_preds.extend(batch_preds)
            all_preds_and_logits.extend(batch_preds_and_logits)
            top_k_labels.extend(batch_top_k_labels)

            all_labels.extend(batch_labels)

        correct_count = defaultdict(int)
        actual_count = defaultdict(int)
        predict_count = defaultdict(int)
        # compute singular class performances and macro performances
        bar = tqdm(desc="computing macro performances", total=len(all_preds))
        for labels, preds in zip(all_labels, all_preds):
            for pred in preds:
                predict_count[pred] += 1

                if pred in labels:
                    correct_count[pred] += 1
            
            for label in labels:
                actual_count[label] += 1
            bar.update(1)
        bar.close()

        def compute_f1(p, r):
            return (2*p*r)/(p + r) if p + r else 0

        precisions = {k: correct_count[k]/predict_count[k] if predict_count[k] else 0 for k in label2id.keys()}
        recalls = {k: correct_count[k]/actual_count[k] if actual_count[k] else 0 for k in label2id.keys()}
        f1s = {k: compute_f1(precisions[k], recalls[k]) for k in label2id.keys()}

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
            if len(preds) > 0:
                ma_e_precisions.append(correct_preds/len(preds))
            else:
                ma_e_precisions.append(0)
            ma_e_recalls.append(correct_preds / len(labels))
            bar.update(1)
        bar.close()

        macro_example_p = np.mean(ma_e_precisions)
        macro_example_r = np.mean(ma_e_recalls)
        macro_example_f1 = compute_f1(macro_example_p, macro_example_r)
        
        macro_examples['p'][d].append(macro_example_p)
        macro_examples['r'][d].append(macro_example_r)
        macro_examples['f1'][d].append(macro_example_f1)
        
        #compute micro performances
        micro_correct_counter = 0
        micro_true_counter = 0
        micro_pred_counter = 0

        bar = tqdm(desc="computing micro performances", total=len(all_preds))       
        for labels, preds in zip(all_labels, all_preds):
            micro_true_counter += len(labels)
            micro_pred_counter += len(preds)
            correct_preds = len(set(labels).intersection(set(preds)))
            micro_correct_counter += correct_preds
            bar.update(1)
        bar.close()
        micro_p = micro_correct_counter/micro_pred_counter if micro_pred_counter else 0
        micro_r = micro_correct_counter/micro_true_counter
        micro_f1 = compute_f1(micro_p, micro_r)

        micros['p'][d].append(micro_p)
        micros['r'][d].append(micro_r)
        micros['f1'][d].append(micro_f1)

        with open(dataset_paths[dataset_id], 'r') as inp:
            lines = [json.loads(l) for l in inp.readlines()]

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

        with open(prediction_file + '_' + d + '.txt', 'a') as out:
            out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('label_#', 'precision', 
                                                                            'recall', 'f1', 'sentence', 'mention', 
                                                                            'preds_and_logits', 'top_k_labels_and_logits', 'true_labels'))
            for label in ordered_labels:
                i = 0
                for sentence, mention, preds_and_logits, top_k, true_label in label_sentences[label]:
                    out_string = '{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{}\t{}\n'.format(label + '_' + str(i + 1),
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
        with open(performance_file + '_' + d + '.txt', 'a') as out:
            out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('macro_examples_p', 'macro_examples_r', 'macro_examples_f1',
                                                                    'macro_p','macro_r', 'macro_f1',
                                                                    'micro_p', 'micro_r', 'micro_f1'))
            out.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(macro_example_p,
                                                                                                        macro_example_r,
                                                                                                        macro_example_f1,
                                                                                                        macro_p,
                                                                                                        macro_r,
                                                                                                        macro_f1,
                                                                                                        micro_p,
                                                                                                        micro_r,
                                                                                                        micro_f1))
                                                                                                        



name = {
    "p": "precision",
    "r": "recall",
    "f1": "f1"
}
for d in keys:
    results = {}
    for result_name, result in zip(["micro", "macro", "example"],
                                    [ micros,  macros, macro_examples]):
        print(result_name)
        print(result)
        print()
        for k, v in result.items():
            v = np.array(v[d])
            mu = np.mean(v)
            sd = np.std(v)
            results["{}_{}".format(result_name, k)] = (mu, sd)

    with open(average_std_file + '_'+ d + '.txt', 'a') as out:
        # out.write('{:^40}\n'.format('-'))
        out.write("model,mu,sd\n")
        for k, (m, s) in results.items():
            out.write('{},{:.4f},{:.4f}\n'.format(k, m, s))
        out.write('\n')
