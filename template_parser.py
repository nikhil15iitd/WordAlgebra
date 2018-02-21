import numpy as np
import json, os, re
from predict_template import read_draw_template, read_unique_templates

# Global variables for length of inputs & outputs
PROBLEM_LENGTH = 105
TEMPLATE_LENGTH = 30
SEED = 23
COEFFS = ['a', 'b', 'c', 'd', 'e', 'f']
UNKOWNS = ['m', 'n']
SLOTS = ['m', 'n', 'a', 'b', 'c', 'd', 'e', 'f'] # unknowns + coeffs

# the index where the vector starts listing coeff values
# 3 because we are using a representation like this: [ template_index, m,n, a,b,c,d,e,f ]
COEFF_OFFSET = 3
UNK_OFFSET = 1

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
    print(templates) # => [a * m + b * m = b * c - a * c, ...]

    for template in templates:
        # this code assumes that coeffs only contains numbers

        # 1. replace coeffs in the template with values:
        for i, coeff in enumerate(COEFFS):
            #print(coeff)
            #print(derivation[i+COEFF_OFFSET])
            if derivation[i+COEFF_OFFSET] == 0:
                continue

            #template = template.replace(coeff, str(derivation[i+COEFF_OFFSET]))
            template = template.replace(coeff, vocab[derivation[i+COEFF_OFFSET]])
            #print(template)

        # 2. replace unknowns in the template with values:
        #for i, unk in enumerate(UNKOWNS):
        #    if derivation[i+UNK_OFFSET] == 0:
        #        continue
        #    template = template.replace(unk, vocab[derivation[i+UNK_OFFSET]])
        tokens = template.split()
        print(tokens)
        for token in tokens:
            pass

        equations.append(template) # add the replaced template

    return equations


# 1. 59, 83, 104, 126, 164, 214, 272, 329, 427, 476, 479
# 2. 513, 504, 501, 493, 488, 483, 482, 473
def get_gold_derivations(vocab):
    """
    @param: vocab: A vocabulary dictionary of the entire corpus for mapping string to index. e.g. '2' => 193
    @return: A list of derivation: [ template_index, m,n, a,b,c,d,e,f ]
    """
    filepath = '0.7 - release/kushman_template_index_fixed.json'
    with open(filepath, 'r') as f:
        dataset = json.load(f)

    derivations = []
    for index, data_sample in enumerate(dataset):
        print('='*50)
        print(index)
        print(data_sample['lEquations'][0])
        template_index = data_sample['template_index']
        templates = data_sample['Template']
        equations = data_sample['lEquations']
        alignments = data_sample['Alignment']

        derivation = []
        derivation.append(template_index)
        m = None
        n = None

        # 1. fill the slots for unknowns
        for i, eq in enumerate(equations):
            #print('='*50)
            #print(eq)

            # https://stackoverflow.com/questions/1059559/split-strings-with-multiple-delimiters
            # Will be splitting on: , <space> - ! ? :
            #tmp = filter(None, re.split("[, \-!?:]+", "Hey, you - what are you doing here!?"))
            instantiated_template = filter(None, re.split("[,\+\-\*\/\=\(\)]+", eq))
            template = list(filter(lambda x: x not in ['+', '-', '*','/', '='], templates[i].split()))
            if index == 513:
                print(instantiated_template)
                print(template)
            for j, symbol in enumerate(template):
                if symbol == 'm':
                    m = instantiated_template[j]
                elif symbol == 'n':
                    n = instantiated_template[j]

        #############################################################
        # TODO: Use your vocab to convert string to word index here
        #############################################################
        #derivation.append(vocab[m])
        #derivation.append(vocab[n])
        derivation.append(m)
        derivation.append(n)


        # 2. fill the slots for coefficients
        existing_coeffs = [a['coeff'] for a in alignments]
        #print(existing_coeffs)
        #for coeff in COEFFS:
        #    if coeff in existing_coeffs:
        #        derivation.append()

        coeff_to_index = {
            'a': 3,
            'b': 4,
            'c': 5,
            'd': 6,
            'e': 7,
            'f': 8,
        }
        # init with zero
        for coeff in COEFFS:
            derivation.append(0)
        for a in alignments:
            #print(a['coeff'])
            derivation[ coeff_to_index[a['coeff']] ] = a['TokenId']#a['Value']
        print(derivation)
        derivations.append(derivation)


    return derivations



def validate_derivation(derivation, dataset):
    """
    Given a vector of derivation (a list of template index + alignments),
    check if it completely matches with the ground truth y.
    """

    #template_index = derivation[0]
    pass


def debug():
    #dataset = read_draw_template('draw_templateindex.json')
    #filepath = '0.7 - release/draw_template_index.json'
    filepath = '0.7 - release/kushman_template_index_fixed.json'
    with open(filepath, 'r') as f:
        dataset = json.load(f)

    pseudo_vocab = {}
    get_gold_derivations(pseudo_vocab)

    '''
    pseudo_vocab = { 1: '9', 2: '12', 3: '15', 4: 'pens', 5: 'students' }
    templates = read_unique_templates(filepath)
    #deriv = [1, 'x', 'y', 9, 12, 15, 0, 0, 0] # [ template_index, m,n, a,b,c,d,e,f ]
    deriv = [1, 4, 5, 1, 2, 3, 0, 0, 0] # [ template_index, m,n, a,b,c,d,e,f ]
    pred_y = derivation_to_y(deriv, templates, pseudo_vocab)
    print(pred_y)
    '''




if __name__ == "__main__":
    debug()