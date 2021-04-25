from collections import defaultdict

def import_bbn_mappings():
    bbn_mappings = {'FIGER': defaultdict(list), 'choi': defaultdict(list), 'onto': defaultdict(list)}

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/BBN_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')
            if splitted[1] != '-':
                bbn_mappings['onto'][splitted[0]].append(splitted[1])
            if splitted[2] != '-':
                bbn_mappings['FIGER'][splitted[0]].append(splitted[2])
            if splitted[3] != '-':
                bbn_mappings['choi'][splitted[0]].append(splitted[3])
    return bbn_mappings

def import_ontonotes_mappings():
    ontonotes_mappings = {'FIGER': defaultdict(list), 'choi': defaultdict(list), 'bbn': defaultdict(list)}

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/OntoNotes_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')
            if splitted[1] != '-':
                ontonotes_mappings['bbn'][splitted[0]].append(splitted[1])
            if splitted[2] != '-':
                ontonotes_mappings['FIGER'][splitted[0]].append(splitted[2])
            if splitted[3] != '-':
                ontonotes_mappings['choi'][splitted[0]].append(splitted[3])
    return ontonotes_mappings

def import_figer_mappings():
    figer_mappings = {'onto': defaultdict(list), 'choi': defaultdict(list), 'bbn': defaultdict(list)}

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/FIGER_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')
            if splitted[1] != '-':
                figer_mappings['bbn'][splitted[0]].append(splitted[1])
            if splitted[2] != '-':
                figer_mappings['onto'][splitted[0]].append(splitted[2])
            if splitted[3] != '-':
                figer_mappings['choi'][splitted[0]].append(splitted[3])
    return figer_mappings

def import_choi_mappings():
    choi_mappings = {'onto': defaultdict(list), 'figer': defaultdict(list), 'bbn': defaultdict(list)}

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/BBN_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')
            if splitted[3] != '-':
                choi_mappings['bbn'][splitted[3]].append(splitted[0])

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/FIGER_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')            
            if splitted[3] != '-':
                choi_mappings['figer'][splitted[3]].append(splitted[0])

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/OntoNotes_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')
            if splitted[3] != '-':
                choi_mappings['onto'][splitted[3]].append(splitted[0])
                
    return choi_mappings