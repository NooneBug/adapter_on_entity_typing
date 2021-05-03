from collections import defaultdict

def import_bbn_mappings():
    bbn_mappings = {'FIGER': defaultdict(list), 'Choi': defaultdict(list), 'OntoNotes': defaultdict(list)}

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/BBN_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')
            if splitted[1] != '-':
                bbn_mappings['OntoNotes'][splitted[0]].append(splitted[1])
            if splitted[2] != '-':
                bbn_mappings['FIGER'][splitted[0]].append(splitted[2])
            if splitted[3] != '-':
                bbn_mappings['Choi'][splitted[0]].append(splitted[3])
    return bbn_mappings

def import_ontonotes_mappings():
    ontonotes_mappings = {'FIGER': defaultdict(list), 'Choi': defaultdict(list), 'BBN': defaultdict(list)}

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/OntoNotes_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')
            if splitted[1] != '-':
                ontonotes_mappings['BBN'][splitted[0]].append(splitted[1])
            if splitted[2] != '-':
                ontonotes_mappings['FIGER'][splitted[0]].append(splitted[2])
            if splitted[3] != '-':
                ontonotes_mappings['Choi'][splitted[0]].append(splitted[3])
    return ontonotes_mappings

def import_figer_mappings():
    figer_mappings = {'OntoNotes': defaultdict(list), 'Choi': defaultdict(list), 'BBN': defaultdict(list)}

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/FIGER_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')
            if splitted[1] != '-':
                figer_mappings['BBN'][splitted[0]].append(splitted[1])
            if splitted[2] != '-':
                figer_mappings['OntoNotes'][splitted[0]].append(splitted[2])
            if splitted[3] != '-':
                figer_mappings['Choi'][splitted[0]].append(splitted[3])
    return figer_mappings

def import_choi_mappings():
    choi_mappings = {'OntoNotes': defaultdict(list), 'FIGER': defaultdict(list), 'BBN': defaultdict(list)}

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/BBN_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')
            if splitted[3] != '-':
                choi_mappings['BBN'][splitted[3]].append(splitted[0])

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/FIGER_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')            
            if splitted[3] != '-':
                choi_mappings['FIGER'][splitted[3]].append(splitted[0])

    with open('/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/OntoNotes_mappings.csv', 'r') as inp:
        lines = [l.replace('\n', '') for l in inp.readlines()]
        for l in lines[1:]:
            splitted = l.split(',')
            if splitted[3] != '-':
                choi_mappings['OntoNotes'][splitted[3]].append(splitted[0])
                
    return choi_mappings
