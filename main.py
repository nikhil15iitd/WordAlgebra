import numpy as np
import json
from nltk.corpus import stopwords, wordnet as wn, words
from nltk.tokenize import word_tokenize
from keras.layers import LSTM, GRU, Conv1D, Dense, PReLU, Dropout, Input, Activation, TimeDistributed
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

# Global variables for length of inputs & outputs
PROBLEM_LENGTH = 50
TEMPLATE_LENGTH = 20

stop_words = set(stopwords.words('english'))
vocab = {}
# nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
operators = {'+': 1, '-': 2, '*': 3, '/': 4, '=': 5}
knowns = {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10, 'f': 11, 'g': 12}
unknowns = {'m': 13, 'n': 14, 'l': 15, 'o': 16, 'p': 17, 'q': 18}


def read_draw():
    X = []
    Y = []
    with open('0.7 - release/draw.json', 'r') as f:
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
            X.append(x)
            Y.append(y)
    return X, Y


def model(input_shape):
    inputs = Input(shape=input_shape)
    l0 = Conv1D(64, 5)(inputs)
    l0 = PReLU()(l0)
    l0 = Conv1D
