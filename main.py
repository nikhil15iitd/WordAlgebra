import numpy as np
import json
from collections import OrderedDict
from nltk.corpus import stopwords, wordnet as wn, words
from nltk.tokenize import word_tokenize
from keras.layers import Bidirectional, LSTM, Conv1D, Dense, PReLU, Dropout, Input, Activation, TimeDistributed
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

# Global variables for length of inputs & outputs
PROBLEM_LENGTH = 105
TEMPLATE_LENGTH = 30

stop_words = set(stopwords.words('english'))
vocab = OrderedDict()
# nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
operators = {'+': 1, '-': 2, '*': 3, '/': 4, '=': 5}
knowns = {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10, 'f': 11, 'g': 12}
unknowns = {'m': 13, 'n': 14, 'l': 15, 'o': 16, 'p': 17, 'q': 18}

'''
Function to map network input (which is numbers back to word problem)
'''


def numbers_to_words(num_vector):
    keys = list(vocab.keys())
    return [keys[i - 1] for i in num_vector]


'''
Function to map network output (which is numbers back to template equation)
'''


def derivation_to_equation(num_vector):
    pass


def read_draw():
    X = []
    Y = []
    max_len = -10
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
            # print(x)
            if max_len < len(x):
                max_len = len(x)
            X.append(x)
            Y.append(y)
    print('Max length: ' + str(max_len))
    return X, Y


def pad_lengths_to_constant(X, Y):
    newX = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=PROBLEM_LENGTH)
    newY = pad_sequences(Y, padding='post', truncating='post', value=0., maxlen=TEMPLATE_LENGTH)
    return newX, newY


'''
Deep neural network model to get F(x) which is to be fed to SPENs
'''


def model(input_shape):
    inputs = Input(shape=input_shape)
    l0 = Bidirectional(LSTM(32, return_sequences=True))(inputs)
    l0 = Bidirectional(LSTM(32, return_sequences=True))(l0)
    model = Model(inputs=inputs, outputs=l0)


def main():
    X, Y = pad_lengths_to_constant(read_draw())
