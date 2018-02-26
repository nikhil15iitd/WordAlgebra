from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import keras
from keras.layers import Bidirectional, LSTM, Dense, PReLU, MaxPool1D, Input, add, TimeDistributed, Activation, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import json
import numpy as np

# Global variables for length of inputs & outputs
PROBLEM_LENGTH = 105
TEMPLATE_LENGTH = 30
SEED = 23

stop_words = set(stopwords.words('english'))
vocab = OrderedDict()

def read_unique_templates(filepath):
    unique_templates = []

    with open(filepath, 'r') as f:
        datastore = json.load(f)
    for data_sample in datastore:
        template = data_sample['Template']
        if not template in unique_templates:
            unique_templates.append(template)

    return unique_templates


def read_draw_template(filepath):
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

            if max_len < len(x):
                max_len = len(x)
            X.append(x)
            Y.append( questions['template_index'] )
    print('Max length: ' + str(max_len))
    return X, Y


def feed_forward_model(input_shape):
    '''
    Deep neural network model to get F(x) which is to be fed to SPENs
    '''
    print(input_shape)
    inputs = Input(shape=input_shape)

    model = Sequential()
    model.add( Bidirectional(LSTM(32, return_sequences=False), input_shape=input_shape,) )
    #print(model.outputs)
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(230)) # num_classes=230
    model.add(Activation('softmax'))

    print(model.outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def pad_lengths_to_constant(X):
    newX = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=PROBLEM_LENGTH)
    newX = np.reshape(newX, (-1, PROBLEM_LENGTH, 1))
    return newX

def main():
    X, Y = read_draw_template('0.7 - release/draw_template_index.json')
    print(len(X))
    print(len(Y))
    print(Y[:10])
    X = pad_lengths_to_constant(X)
    F = feed_forward_model(X.shape[1:])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    print(X_train.shape)
    print(len(y_train))
    print(X_test.shape)
    print(len(y_test))
    y_train = keras.utils.to_categorical(y_train, num_classes=230)
    y_test = keras.utils.to_categorical(y_test, num_classes=230)

    F.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test))
    y_pred = F.predict(X_test)
    for i in range(y_pred.shape[0]):
        print(derivation_to_equation(y_pred[i].reshape((TEMPLATE_LENGTH,))))
        print(derivation_to_equation(y_test[i].reshape((TEMPLATE_LENGTH,))))



if __name__ == "__main__":
    main()