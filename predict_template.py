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
import os
import globals

#from main import load_glove
from template_parser import debug

# Global variables for length of inputs & outputs
PROBLEM_LENGTH = 105
TEMPLATE_LENGTH = 30
SEED = 23
GLOVE_DIR = 'glove.6B'
EMBEDDING_DIM = 50  # 50
MAX_SEQUENCE_LENGTH = 105

stop_words = set(stopwords.words('english'))
vocab = OrderedDict()

def load_glove(vocab):
    # ref: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    vocab_size = len(vocab.keys())
    word_index = vocab_size

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.%dd.txt' % EMBEDDING_DIM), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    # embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    embedding_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM))

    # for word, i in word_index.items():
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    from keras.layers import Embedding

    # embedding_layer = Embedding(len(word_index) + 1,
    embedding_layer = Embedding(vocab_size + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    return embedding_layer

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


def feed_forward_model(input_shape, emb_layer):
    '''
    Deep neural network model to get F(x) which is to be fed to SPENs
    '''
    print(input_shape)
    inputs = Input(shape=input_shape)
    emb = emb_layer(inputs)

    #l = Bidirectional(LSTM(32, return_sequences=True))(emb)
    l0 = Bidirectional(LSTM(16, return_sequences=True))(emb)
    print(l0.shape)
    l0 = Bidirectional(LSTM(32))(l0)
    print(l0.shape)
    #l = TimeDistributed(Dense(32, activation='relu'), name='seq')(l0)
    l = Dense(32, activation='relu')(l0)
    print(l.shape)
    #fds
    l = Dense(TEMPLATE_LENGTH, activation='softmax')(l)

    model = Model(inputs=inputs, outputs=l)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def pad_lengths_to_constant(X):
    newX = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=PROBLEM_LENGTH)
    newX = np.reshape(newX, (-1, PROBLEM_LENGTH, 1))
    return newX

def main():
    derivations, vocab_dataset = debug()
    #X, Y = read_draw_template('0.7 - release/draw_template_index.json')
    X, Xtags, YSeq, Y, Z = derivations

    emb_layer = load_glove(vocab_dataset)
    #X = pad_lengths_to_constant(X)
    #X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=globals.PROBLEM_LENGTH)
    X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=globals.PROBLEM_LENGTH)
    X = np.array(X)
    print(X.shape)
    Y = Y[:, 0].reshape(-1, 1)
    print(Y.shape)


    #F = feed_forward_model(X.shape[1:], emb_layer)
    F = feed_forward_model(X.shape[1:], emb_layer)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    print(X_train.shape)
    print(len(y_train))
    print(X_test.shape)
    print(len(y_test))
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    #y_train = keras.utils.to_categorical(y_train, num_classes=230)
    #y_test = keras.utils.to_categorical(y_test, num_classes=230)
    print(y_train.shape)    #(411, 7, 230)
    print(y_test.shape)


    F.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test))
    y_pred = F.predict(X_test)
    for i in range(y_pred.shape[0]):
        print(derivation_to_equation(y_pred[i].reshape((TEMPLATE_LENGTH,))))
        print(derivation_to_equation(y_test[i].reshape((TEMPLATE_LENGTH,))))



if __name__ == "__main__":
    globals.init()  # Fetch global variables such as PROBLEM_LENGTH
    main()