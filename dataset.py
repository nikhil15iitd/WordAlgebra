import json, nltk
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import globals


def numbers_to_words(vocab, num_vector):
    '''
    Function to map network input (which is numbers back to word problem)
    '''
    keys = list(vocab.keys())
    return [keys[i - 1] for i in num_vector]


def derivation_to_equation(num_vector):
    '''
    Function to map network output (which is numbers back to template equation)
    '''
    return [all_template_vars[int(i)] for i in num_vector]

def read_draw(filepath='0.7 - release/draw.json'):
    vocab = OrderedDict()
    # nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
    operators = {'+': 1, '-': 2, '*': 3, '/': 4, '=': 5}
    knowns = {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10, 'f': 11, 'g': 12}
    unknowns = {'m': 13, 'n': 14, 'l': 15, 'o': 16, 'p': 17, 'q': 18}
    symbols = {'%': 19}
    separators = {',': 20}
    all_template_vars = {0: ' ', 20: ','}


    for key in operators.keys():
        all_template_vars[operators[key]] = key
    for key in unknowns.keys():
        all_template_vars[unknowns[key]] = key
    for key in knowns.keys():
        all_template_vars[knowns[key]] = key
    for key in symbols.keys():
        all_template_vars[symbols[key]] = key
    X = []
    Y = []
    max_len = -10
    with open(filepath, 'r') as f:
        datastore = json.load(f)
        for questions in datastore:
            x = []
            y = []
            # process each question
            # split into sentences
            sentences = questions['sQuestion'].split('.')
            for sentence in sentences:
                word_tokens = word_tokenize(sentence)
                for word in word_tokens:
                    if word not in vocab:
                        vocab[word] = len(vocab) + 1
                    x.append(vocab[word])
                x.append(20)
            for template in questions['Template']:
                for slot in template.split(' '):
                    if slot in knowns:
                        y.append(knowns[slot])
                    elif slot in unknowns:
                        y.append(unknowns[slot])
                    elif slot in operators:
                        y.append(operators[slot])
                    else:
                        y.append(0)
                y.append(20)
            # print(y)
            # print(x)
            if max_len < len(x):
                max_len = len(x)
            X.append(x)
            Y.append(y)
    print('Max length: ' + str(max_len))
    return X, Y


if __name__ == "__main__":
    globals.init() # Fetch global variables such as PROBLEM_LENGTH
    #stop_words = set(stopwords.words('english'))
    X, Y = read_draw('0.7 - release/draw.json')
    print(X[0])
    print(Y[0])