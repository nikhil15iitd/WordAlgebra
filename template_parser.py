import numpy as np
import json, os, re
from predict_template import read_draw_template, read_unique_templates

# Global variables for length of inputs & outputs
PROBLEM_LENGTH = 105
TEMPLATE_LENGTH = 30
SEED = 23
COEFFS = ['a', 'b', 'c', 'd', 'e', 'f']
UNKOWNS = ['m', 'n']
SLOTS = ['m', 'n', 'a', 'b', 'c', 'd', 'e', 'f']  # unknowns + coeffs

# the index where the vector starts listing coeff values
# 3 because we are using a representation like this: [ template_index, m,n, a,b,c,d,e,f ]
COEFF_OFFSET = 3
UNK_OFFSET = 1


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def derivation_to_y(derivation, templates, vocab):
    """
    Given a vector of derivation (a list of template index + alignments),
    convert it to a vector of equation indices.

    @param: derivation: e.g. [ template_index, m,n, a,b,c,d,e,f ]
    @param: templates: A list of all templates in the target dataset.
    @param: vocab: vocaburaly dictionary of the entire corpus.
    @return A list of equations represented in strings.
    """
    raise NotImplementedError

    equations = []
    templates = templates[derivation[0]]
    print(templates)  # => [a * m + b * m = b * c - a * c, ...]

    for template in templates:
        # this code assumes that coeffs only contains numbers

        # 1. replace coeffs in the template with values:
        for i, coeff in enumerate(COEFFS):
            # print(coeff)
            # print(derivation[i+COEFF_OFFSET])
            if derivation[i + COEFF_OFFSET] == 0:
                continue

            # template = template.replace(coeff, str(derivation[i+COEFF_OFFSET]))
            template = template.replace(coeff, vocab[derivation[i + COEFF_OFFSET]])
            # print(template)

        # 2. replace unknowns in the template with values:
        # for i, unk in enumerate(UNKOWNS):
        #    if derivation[i+UNK_OFFSET] == 0:
        #        continue
        #    template = template.replace(unk, vocab[derivation[i+UNK_OFFSET]])
        tokens = template.split()
        print(tokens)
        for token in tokens:
            pass

        equations.append(template)  # add the replaced template

    return equations


def get_gold_derivations(dataset, vocab):
    """
    @param: dataset: A dictionary loaded from json file
    @param: vocab: A vocabulary dictionary of the entire corpus for mapping string to index. e.g. '2' => 193
    @return: A list of derivation: [ template_index, m,n, a,b,c,d,e,f ]
    """
    X = []
    derivations = []
    for index, data_sample in enumerate(dataset):
        print('=' * 50)
        print(index)
        words = data_sample['sQuestion'].split()
        x_temp = []
        for w in words:
            x_temp.append(vocab[w])
        X.append(x_temp)
        template_index = data_sample['template_index']
        templates = data_sample['Template']
        equations = data_sample['lEquations']
        alignments = data_sample['Alignment']

        # 1. Add the template index
        tmp = []
        tmp.append(template_index)

        # An attempt to extract unknown string from lEquations:
        '''for i, eq in enumerate(equations):
            # https://stackoverflow.com/questions/1059559/split-strings-with-multiple-delimiters
            # Will be splitting on: , <space> - ! ? :
            instantiated_template = filter(None, re.split("[,\+\-\*\/\=\(\)]+", eq))
            template = list(filter(lambda x: x not in ['+', '-', '*','/', '='], templates[i].split()))
            print(instantiated_template)
            print(template)
            for j, symbol in enumerate(template):
                if symbol == 'm':
                    m = instantiated_template[j]
                elif symbol == 'n':
                    n = instantiated_template[j]
        '''

        # 2. fill the slots
        existing_slots = [a['coeff'] if 'coeff' in a else a['unk'] for a in alignments]
        # print(existing_slots)

        slot_to_index = {
            'm': 1,
            'n': 2,
            'a': 3,
            'b': 4,
            'c': 5,
            'd': 6,
            'e': 7,
            'f': 8,
        }
        # init with zero
        for slot in SLOTS:
            tmp.append(0)
        for a in alignments:
            if 'coeff' in a:
                tmp[slot_to_index[a['coeff']]] = a['Value']
            elif 'unk' in a:
                tmp[slot_to_index[a['unk']]] = a['String']
            else:
                raise
        # print(tmp)

        # 3. Use the vocab to convert string to word index
        derivation = tmp[:]
        print(derivation)
        if not is_number(derivation[1]):
            derivation[1] = vocab[derivation[1].split('_')[0]]
        if not is_number(derivation[2]):
            derivation[2] = vocab[derivation[2].split('_')[0]]
        print(derivation)
        derivations.append(derivation)

    print(derivations)
    return X, derivations


def validate_derivation(derivation, dataset):
    """
    Given a vector of derivation (a list of template index + alignments),
    check if it completely matches with the ground truth y.
    """
    pass


def debug():
    filepath = '0.7 - release/kushman_template_index_debug.json'
    with open(filepath, 'r') as f:
        dataset = json.load(f)

    # Build vocab for question texts
    from collections import defaultdict
    word_count = defaultdict(float)
    for index, data_sample in enumerate(dataset):
        words = data_sample['sQuestion'].split()
        for w in words:
            word_count[w] += 1
    print(word_count)

    word_idx_map = dict()
    idx_word_map = dict()
    for i, word in enumerate(word_count):
        word_idx_map[word] = i
    print(word_idx_map)
    print(len(word_idx_map.keys()))

    return get_gold_derivations(dataset, word_idx_map), word_idx_map


if __name__ == "__main__":
    debug()
