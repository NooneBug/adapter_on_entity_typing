import configparser
from adapter_entity_typing.network_classes.classifiers import adapterPLWrapper
from torch.utils.data.dataloader import DataLoader
from adapter_entity_typing.utils import prepare_entity_typing_datasets
from adapter_entity_typing.network import get_model, add_classifier, load_model
from collections import defaultdict
import torch
import json
import numpy as np


parameter_tags = ['adapters2_bbn']
config = configparser.ConfigParser()
training_config_file = "result_scripts/generate_predictions_parameters.ini"
config.read("result_scripts/generate_predictions_parameters.ini")
print(list(config.keys()))
config = config[parameter_tags[0]]


# model_path = config['ModelRootPath'] + config['ModelName']
# classifier = get_model(model_path)

# max_context_side_size = classifier.configuration('MaxContextSideSize')
# max_entity_size = classifier.configuration('MaxEntitySize')

# train_dataset, dev_dataset, test_dataset, label2id = prepare_entity_typing_datasets(classifier)


# vocab_len = len(id2label)

# add_classifier(model = classifier, labels = label2id)

# model = adapterPLWrapper.load_from_checkpoint(model_path, 
#                                                 adapterClassifier = classifier, 
#                                                 id2label = id2label, 
#                                                 lr = 1e-4)
# model.cuda()
# model.eval()

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
training_name = config['training_name']
performance_file = config['performanceFile'] + config['experiment_name']
prediction_file = config['predictionFile'] + config['experiment_name']
average_std_file = config['AvgStdFile'] + config['experiment_name']


parameter_tags = ['adapters2_bbn']

for model, train_dataset, dev_dataset, test_dataset, label2id in load_model(parameter_tags[0], training_config_file):  # , "results_scripts/generate_preditcions_parameters.ini"):

    dev_loader = DataLoader(dev_dataset, batch_size = 100, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size = 100, num_workers=20)
    id2label = {v: k for k,v in label2id.items()}

    dev_or_test = config('dev_or_test')

    if dev_or_test == 'both':
        data_to_pred = ['dev', 'test']
        datasets = [dev_loader, test_loader]
        dataset_paths = [model.configuration('PathInputDev'), model.configuration('PathInputTest')]

    elif dev_or_test == 'dev':
        data_to_pred = ['dev']
        datasets = [dev_loader]
        dataset_paths = [model.configuration('PathInputDev')]

    elif dev_or_test == 'test':
        data_to_pred = ['test']
        datasets = [test_loader]
        dataset_paths = [model.configuration('PathInputTest')]

    else:
        raise Exception('please provide a meaningfull value for "dev_or_test"')

    for dataset_id, d in enumerate(data_to_pred):
        all_preds = []
        all_preds_and_logits = []
        all_labels = []
        top_k_labels = []
        loader = datasets[dataset_id]
        for mention, attn, labels in loader:
            
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
                    ex_preds.append(top_k_l[0][0])
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
        # compute singular class performances and macro performances
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

        macro_p = np.mean(list(precisions.values()))
        macro_r = np.mean(list(recalls.values()))
        macro_f1 = compute_f1(macro_p, macro_r)

        macros['p'].append(macro_p)
        macros['r'].append(macro_r)
        macros['f1'].append(macro_f1)

        #compute macro_example performances
        ma_e_precisions = []
        ma_e_recalls = []
        n = len(all_labels)

        for labels, preds in zip(all_labels, all_preds):
            correct_preds = len(set(labels).intersection(set(preds)))
            ma_e_precisions.append(correct_preds/len(preds))
            ma_e_recalls.append(correct_preds / len(labels))

        macro_example_p = np.mean(ma_e_precisions)
        macro_example_r = np.mean(ma_e_recalls)
        macro_example_f1 = compute_f1(macro_example_p, macro_example_r)
        
        macro_examples['p'].append(macro_example_p)
        macro_examples['r'].append(macro_example_r)
        macro_examples['f1'].append(macro_example_f1)
        
        #compute micro performances
        micro_correct_counter = 0
        micro_true_counter = 0
        micro_pred_counter = 0

        for labels, preds in zip(all_labels, all_preds):
            micro_true_counter += len(labels)
            micro_pred_counter += len(preds)
            correct_preds = len(set(labels).intersection(set(preds)))
            micro_correct_counter += len(correct_preds)
        
        micro_p = micro_correct_counter/micro_pred_counter
        micro_r = micro_correct_counter/micro_true_counter
        micro_f1 = compute_f1(micro_p, micro_r)

        micros['p'].append(micro_p)
        micros['r'].append(micro_r)
        micros['f1'].append(micro_f1)

        with open(dataset_paths[dataset_id], 'r') as inp:
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

        with open(prediction_file + d + '.txt', 'a') as out:
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
        with open(performance_file + d + '.txt', 'a') as out:
            out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('macro_examples_p', 'macro_examples_r', 'macro_examples_f1'
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
results = {}
for result_name, result in zip(["micro", "macro", "example"],
                                 [ micros,  macros, macro_examples]):
    print(result_name)
    print(result)
    print()
    for k, v in result.items():
        v = np.array(v)
        mu = np.mean(v)
        sd = np.std(v)
        results["{}_{}".format(result_name, k)] = (mu, sd)

with open(average_std_file + '.txt', 'a') as out:
    # out.write('{:^40}\n'.format('-'))
    out.write("model,mu,sd\n")
    for k, (m, s) in results.items():
        out.write('{},{:.4f},{:.4f}\n'.format(k, m, s))
    out.write('\n')
