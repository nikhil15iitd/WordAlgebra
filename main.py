import numpy as np
import json
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.layers import Bidirectional, LSTM, Conv1D, Dense, PReLU, MaxPool1D, Input, add, TimeDistributed
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Global variables for length of inputs & outputs
PROBLEM_LENGTH = 105
TEMPLATE_LENGTH = 30
SEED = 23

stop_words = set(stopwords.words('english'))
vocab = OrderedDict()
# nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
operators = {'+': 1, '-': 2, '*': 3, '/': 4, '=': 5}
knowns = {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10, 'f': 11, 'g': 12}
unknowns = {'m': 13, 'n': 14, 'l': 15, 'o': 16, 'p': 17, 'q': 18}
separators = {',': 20}
all_template_vars = {0: ' ', 20: ','}


def numbers_to_words(num_vector):
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


def read_draw():
    for key in operators.keys():
        all_template_vars[operators[key]] = key
    for key in unknowns.keys():
        all_template_vars[unknowns[key]] = key
    for key in knowns.keys():
        all_template_vars[knowns[key]] = key
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
    newX = np.reshape(newX, (-1, PROBLEM_LENGTH, 1))
    newY = np.reshape(newY, (-1, TEMPLATE_LENGTH, 1))
    return newX, newY


def feed_forward_model(input_shape):
    '''
    Deep neural network model to get F(x) which is to be fed to SPENs
    '''
    inputs = Input(shape=input_shape)
    l0 = Bidirectional(LSTM(32, return_sequences=True))(inputs)
    l0 = Bidirectional(LSTM(32, return_sequences=True))(l0)
    l0 = Conv1D(64, 5)(l0)
    l0 = PReLU()(l0)
    l0 = Conv1D(64, 5)(l0)
    l0 = PReLU()(l0)
    l0 = MaxPool1D(3)(l0)
    l0 = Conv1D(128, 3)(l0)
    l0 = PReLU()(l0)
    l0 = TimeDistributed(Dense(1, activation='relu'))(l0)
    model = Model(inputs=inputs, outputs=l0)
    model.compile('adam', 'mse', metrics=['acc'])
    return model


def main():
    X, Y = read_draw()
    X, Y = pad_lengths_to_constant(X, Y)
    F = feed_forward_model(X.shape[1:])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    F.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))
    y_pred = F.predict(X_test)
    for i in range(y_pred.shape[0]):
        print(derivation_to_equation(y_pred[i].reshape((TEMPLATE_LENGTH,))))
        print(derivation_to_equation(y_test[i].reshape((TEMPLATE_LENGTH,))))


main()
