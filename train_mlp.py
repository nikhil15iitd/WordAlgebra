import numpy as np
import json
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import keras
from keras.layers import Bidirectional, LSTM, Conv1D, Dense, PReLU, MaxPool1D, Input, Embedding, TimeDistributed, \
    BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from template_parser import debug

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
symbols = {'%': 19}
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
    for key in symbols.keys():
        all_template_vars[symbols[key]] = key
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
    newY = np.reshape(newY, (-1, TEMPLATE_LENGTH, 1))
    return newX, newY


def feed_forward_model(input_shape, vocab_size, template_vocab_size):
    '''
    Deep neural network model to get F(x) which is to be fed to SPENs
    '''
    inputs = Input(shape=input_shape)
    l0 = Embedding(vocab_size, 16, input_length=PROBLEM_LENGTH)(inputs)
    l0 = Bidirectional(LSTM(32, return_sequences=True))(l0)
    l0 = Bidirectional(LSTM(32, return_sequences=True))(l0)
    l0 = Conv1D(64, 5)(l0)
    l0 = PReLU()(l0)
    l0 = Conv1D(64, 5)(l0)
    l0 = PReLU()(l0)
    l0 = MaxPool1D(3)(l0)
    l0 = Conv1D(128, 3)(l0)
    l0 = PReLU()(l0)
    l0 = TimeDistributed(Dense(template_vocab_size, activation='softmax'))(l0)
    model = Model(inputs=inputs, outputs=l0)
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
    return model


def feed_forward_mlp_model(input_shape, vocab_size):
    '''
    Deep neural network model to get F(x) which is to be fed to SPENs
    '''
    print(input_shape) # 105,
    print(vocab_size) # => 183
    print(keras.__version__)
    num_decoder_tokens = 9 # since the template vector has 9 elems
    inputs = Input(shape=input_shape)
    emb = Embedding(vocab_size, 16, input_length=PROBLEM_LENGTH)(inputs) # => (?, 105, 16)



    '''
    # v1: softmax ver
    #l0 = keras.layers.GRU(32, return_sequences=True)(emb)
    l0 = keras.layers.GRU(32, return_sequences=False)(emb) # => (?, 32)
    l0 = Dense(num_decoder_tokens * vocab_size, activation='softmax')(l0)
    #l0 = Dense(8, activation='softmax')(l0)
    print(l0.shape)
    l0 = keras.layers.Reshape((num_decoder_tokens, vocab_size))(l0)
    l0 = TimeDistributed(Dense(num_decoder_tokens, activation='softmax'))(l0)
    print('output shape:')
    print(l0.shape)
    '''

    '''
    '''
    # v1: linear act ver
    l0 = keras.layers.GRU(32, return_sequences=False)(emb) # => (?, 32)

    l0 = keras.layers.RepeatVector(num_decoder_tokens)(l0)
    l0 = keras.layers.GRU(32, return_sequences=True)(l0) # => (?, 32)
    #l0 = Dense(num_decoder_tokens, activation='relu')(l0)
    #l0 = keras.layers.Reshape((num_decoder_tokens, 1))(l0) # change "1" to vocab_size for categorical loss approach
    #l0 = TimeDistributed(Dense(num_decoder_tokens, activation='softmax'))(l0)

    #l0 = TimeDistributed(Dense(num_decoder_tokens, activation='linear'))(l0)
    l0 = TimeDistributed(Dense(1, activation='linear'))(l0)
    #l0 = Dense(num_decoder_tokens, activation='linear')(l0)
    print('output shape:')
    print(l0.shape)


    # v2: enc-dec
    '''
    encoder = keras.layers.GRU(32, return_state=True)
    enc_outs, enc_h = encoder(emb) # enc_h: encoder states

    dec_inp = Input(shape=(None, num_decoder_tokens))
    decoder = keras.layers.GRU(32, return_sequences=True, return_state=True)
    #decoder = keras.layers.GRU(32, return_state=True)

    #dec_outs, _ = decoder(dec_inp, initial_state=enc_h)
    dec_outs, _ = decoder(enc_outs, initial_state=enc_h)
    print(dec_outs.shape)
    l0 = dec_outs
    #dense = Dense(num_decoder_tokens * vocab_size, activation='softmax')(enc_outs)

    #model.add(Dense(hidden_neurons, in_out_neurons))
    #dense = Dense(32, vocab_size, activation='softmax')(enc_outs)
    #print(dense.shape)
    #dense = keras.layers.Reshape((num_decoder_tokens, vocab_size))(dense)

    l0 = TimeDistributed(Dense(num_decoder_tokens, activation='softmax'))(l0)
    print('output shape:')
    print(l0.shape)
    '''


    #model = Model(inputs=inputs, outputs=outputs) # => (?, 105, 8)
    #model = Model(inputs=[inputs, dec_inp], outputs=dec_outs)
    model = Model(inputs=inputs, outputs=l0)
    #model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
    model.compile('adam', 'mse', metrics=['acc'])
    return model


def get_layers():
    layers = [(1000, 'relu')]
    enlayers = [(250, 'softplus')]
    return (layers, enlayers)


def main():
    '''
    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    '''


    (X, Y), vocab_dataset = debug()
    print('#'*100)
    print(X[0])
    X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=PROBLEM_LENGTH)
    print('#'*100)
    print(Y)
    Y = np.array(Y)


    F = feed_forward_mlp_model(X.shape[1:], len(vocab_dataset.keys()))
    # X, Y = read_draw()
    # X, Y = pad_lengths_to_constant(X, Y)
    # F = feed_forward_model(X.shape[1:], len(vocab.keys()), len(all_template_vars.keys()))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    ntrain = X_train.shape[0]


    # (convert y to one-hot in the case of categorical losses)
    #y_train = keras.utils.to_categorical(y_train)
    #y_test = keras.utils.to_categorical(y_test)
    y_train = y_train.reshape(-1, 9, 1)
    y_test = y_test.reshape(-1, 9, 1)

    F.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))
    y_pred = np.argmax(F.predict(X_test), axis=2)
    for i in range(y_pred.shape[0]):
        print('#'*100)
        print(derivation_to_equation(y_pred[i].reshape((TEMPLATE_LENGTH,))))
        print(derivation_to_equation(y_test[i].reshape((TEMPLATE_LENGTH,))))




main()
