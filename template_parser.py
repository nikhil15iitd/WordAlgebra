import numpy as np
import json
import nltk
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import defaultdict

# Global variables for length of inputs & outputs
PROBLEM_LENGTH = 105
TEMPLATE_LENGTH = 30
SEED = 23
COEFFS = ['a', 'b', 'c', 'd', 'e', 'f']
UNKOWNS = ['m', 'n']
SLOTS = ['a', 'b', 'c', 'd', 'e', 'f']  # unknowns + coeffs
vocab_template = {' ': 0, 'm': 1, 'n': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, '0.01': 9, '*': 10, '+': 11,
                  '-': 12, '=': 13, ',': 14, '0': 15, '1': 16, '2': 17, '3': 18, '4': 19, '5': 20}
template_vocab_size = len(list(vocab_template.keys()))

operators = {'+': 1, '-': 2, '*': 3, '/': 4, '=': 5}
knowns = {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10, 'f': 11, 'g': 12}
unknowns = {'m': 13, 'n': 14, 'l': 15, 'o': 16, 'p': 17, 'q': 18}
symbols = {'%': 19}
separators = {',': 20}
all_template_vars = {0: ' ', 20: ','}

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
    Xtags = []
    YSeq = []
    derivations = []
    unique_templates = []
    solutions = []
    tags_to_int = dict()
    tag_count = 1
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for index, data_sample in enumerate(dataset):
        #print('=' * 50)
        #print(index)
        sentence_list = tokenizer.tokenize(data_sample['sQuestion'])
        cum_len = np.zeros(len(sentence_list))
        for i in range(1, len(sentence_list)):
            cum_len[i] = cum_len[i - 1] + len(sentence_list[i - 1].split())

        words = nltk.word_tokenize(data_sample['sQuestion'])
        tags = nltk.pos_tag(words)
        tags = [x[1] for x in tags]
        x_temp = []
        for tag in tags:
            if tag not in tags_to_int:
                tags_to_int[tag] = tag_count
                tag_count += 1
            x_temp.append(tags_to_int[tag])
        Xtags.append(x_temp)
        x_temp = []
        for w in words:
            x_temp.append(vocab[w.lower()])
        X.append(x_temp)
        if data_sample["Template"] not in unique_templates:
            unique_templates.append(data_sample["Template"])
        template_index = unique_templates.index(data_sample["Template"])
        templates = data_sample['Template']
        equations = data_sample['lEquations']
        alignments = data_sample['Alignment']

        # create template sequences
        x_temp = []
        for template in data_sample['Template']:
            template_tokens = template.split()
            for token in template_tokens:
                x_temp.append(vocab_template[token])
            x_temp.append(vocab_template[' '])
        YSeq.append(np.array(x_temp))

        # 1. Add the template index
        tmp = []
        tmp.append(template_index)

        # 2. fill the slots
        existing_slots = [a['coeff'] if 'coeff' in a else a['unk'] for a in alignments]
        # print(existing_slots)

        slot_to_index = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5,
            'f': 6
        }
        # init with zero
        for slot in SLOTS:
            tmp.append(0)
        for a in alignments:
            if 'coeff' in a:
                tmp[slot_to_index[a['coeff']]] = cum_len[a['SentenceId']] + a['TokenId'] + 1

        derivations.append(np.array(tmp))
        solutions.append(np.array(data_sample['lSolutions']))
    print(unique_templates)
    print(tags_to_int)
    print('Number of unique tags: ' + str(len(tags_to_int.keys())))
    return X, Xtags, np.array(YSeq), np.array(derivations), np.array(solutions)


def validate_derivation(derivation, dataset):
    """
    Given a vector of derivation (a list of template index + alignments),
    check if it completely matches with the ground truth y.
    """
    pass


def debug(verbose=False):
    # filepath = '0.7 - release/kushman_template_index_debug.json'
    # filepath = '0.7 - release/kushman_template_index_org.json'
    filepath = '0.7 - release/kushman.json'
    with open(filepath, 'r') as f:
        dataset = json.load(f)


    # Build vocab for question texts
    word_count = defaultdict(float)

    for index, data_sample in enumerate(dataset):
        words = nltk.word_tokenize(data_sample['sQuestion'])
        for w in words:
            word_count[w.lower()] += 1

    word_idx_map = dict()
    idx_word_map = dict()
    for i, word in enumerate(word_count):
        word_idx_map[word] = i + 1
    word_idx_map[' '] = 0

    if verbose:
        # print(len(list(dataset.keys()) ))
        print(len(dataset))
        print(word_count)
        print(word_idx_map)
        print(len(word_idx_map.keys()))
        print('#' * 100)

    return get_gold_derivations(dataset, word_idx_map), word_idx_map


if __name__ == "__main__":
    debug()
