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

import config_wordalgebra
from dataset import read_draw, numbers_to_words, derivation_to_equation
from template_parser import debug




def feed_forward_rnn_model(input_shape, vocab_size):
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

def custom_softmax(t):
    """
    https://datascience.stackexchange.com/questions/23614/keras-multiple-softmax-in-last-layer-possible
    """
    from keras import backend as K
    sh = K.shape(t)
    partial_sm = []
    for i in range(sh[1] // 4):
        partial_sm.append(K.softmax(t[:, i*4:(i+1)*4]))
    return K.concatenate(partial_sm)


def feed_forward_mlp_model(batch_size, input_shape, vocab_size):
    '''
    Deep neural network model to get F(x) which is to be fed to SPENs
    '''
    print(input_shape) # 105,
    print(vocab_size) # => 183
    print(keras.__version__)
    num_output = 9 # since the template vector has 9 elems
    num_classes = 10 # debug

    inputs = Input(shape=input_shape)
    emb = Embedding(vocab_size, 16, input_length=PROBLEM_LENGTH)(inputs) # => (?, 105, 16)

    l0 = keras.layers.GRU(32, return_sequences=False)(emb) # => (?, 32)
    print(type(l0)) # => <class 'tensorflow.python.framework.ops.Tensor'>

    # 9 Dense layers for predicting word index in each slot
    #l0 = Dense(9, activation=custom_softmax)(l0)
    outputs = []
    for i in range(num_output):
        #outputs.append( Dense(num_output, activation='softmax')(l0) )
        if i == 0:
            outputs.append( Dense(230, activation='softmax')(l0) ) # template
        elif i == 1 or i == 2:
            outputs.append( Dense(vocab_size, activation='softmax')(l0) ) # unknowns
        elif i > 2 and i < 10:
            outputs.append( Dense(PROBLEM_LENGTH, activation='softmax')(l0) ) # coeffs
    print('output shape:')
    print(outputs[0].shape) # => batch_size x 9

    '''
    # construct the output vector based on the prediction result from each softmax
    #from keras import backend as K
    import tensorflow as tf
    #output = tf.zeros([batch_size, num_output], tf.int32)
    output = np.zeros((batch_size, num_output))
    print(output.shape)

    for i in range(num_output):
        output[:, i] = list_out[:, i]
    l0 = tf.convert_to_tensor(output, np.int32)
    '''


    model = Model(inputs=inputs, outputs=outputs)
    #model = Model(inputs=inputs, outputs=[ outputs[0], outputs[1] ])
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])

    return model


def feed_forward_mlp_model_coeffs(batch_size, input_shape, vocab_size):
    '''
    Deep neural network model to get F(x) which is to be fed to SPENs
    '''
    print(input_shape) # 105,
    print(vocab_size) # => 183
    print(keras.__version__)
    num_output = 7 # since the template vector has 7 elems (without unknowns)

    inputs = Input(shape=input_shape)
    emb = Embedding(vocab_size, 16, input_length=config_wordalgebra.PROBLEM_LENGTH)(inputs) # => (?, 105, 16)

    l0 = keras.layers.GRU(32, return_sequences=False)(emb) # => (?, 32)
    print(type(l0)) # => <class 'tensorflow.python.framework.ops.Tensor'>

    # 9 Dense layers for predicting word index in each slot
    #l0 = Dense(9, activation=custom_softmax)(l0)
    outputs = []
    for i in range(num_output):
        if i == 0:
            outputs.append( Dense(230, activation='softmax')(l0) ) # template
        elif i > 0 and i < 8:
            outputs.append( Dense(config_wordalgebra.PROBLEM_LENGTH, activation='softmax')(l0) ) # coeffs
    print('output shape:')
    print(outputs[0].shape) # => batch_size x 9

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer, 'sparse_categorical_crossentropy', metrics=['acc'])

    return model



def get_layers():
    layers = [(1000, 'relu')]
    enlayers = [(250, 'softplus')]
    return (layers, enlayers)


def main():
    #(X, Y), vocab_dataset = debug()
    X, Y, vocab_dataset = debug()
    print('#'*100)
    print(X[0])
    X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=config_wordalgebra.PROBLEM_LENGTH)
    print('#'*100)
    print(Y)
    Y = np.array(Y)

    batch_size = 128
    F = feed_forward_mlp_model_coeffs(batch_size, X.shape[1:], len(vocab_dataset.keys()))
    F.summary()
    # X, Y = read_draw()
    # X, Y = pad_lengths_to_constant(X, Y)
    # F = feed_forward_model(X.shape[1:], len(vocab.keys()), len(all_template_vars.keys()))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=config_wordalgebra.SEED)
    ntrain = X_train.shape[0]
    print(X_train.shape)
    print(X_test.shape)


    template_size = 7

    targets = []
    for i in range(template_size):
        targets.append(y_train[:,i])

    test_targets = []
    for i in range(template_size):
        test_targets.append(y_test[:,i])


    F.fit(X_train, targets, batch_size=batch_size, epochs=100, validation_data=(X_test, test_targets))
    F.save('baseline_debug.h5')
    print('='*100)
    print(F.predict(X_test)[0].shape) # 2, 230
    print(F.predict(X_test)[1].shape) # 2, 183
    pred_train = F.predict(X_test)

    preds = []
    for out in pred_train:
        tmp = np.argmax(out, axis=1)
        preds.append(tmp)
    preds = np.array(preds)
    # Convert the output back to (N x temlpate_size)
    preds = preds.reshape(-1, template_size)
    print(preds.shape)
    y_pred_test = preds

    for i in range(y_pred_test.shape[0]):
        print('='*100)
        #print(derivation_to_equation(y_pred[i].reshape((TEMPLATE_LENGTH,))))
        #print(derivation_to_equation(y_test[i].reshape((TEMPLATE_LENGTH,))))
        #print(derivation_to_equation(y_pred[i]))
        #print(derivation_to_equation(y_test[i]))
        print(y_pred_test[i])
        print(y_test[i])

    print('#'*100)
    print('#'*100)

    for i in range(y_pred_test.shape[0]):
        print('='*100)
        #print(derivation_to_equation(y_pred[i].reshape((TEMPLATE_LENGTH,))))
        #print(derivation_to_equation(y_test[i].reshape((TEMPLATE_LENGTH,))))
        #print(derivation_to_equation(y_pred[i]))
        #print(derivation_to_equation(y_test[i]))
        print(y_pred_test[i])
        print(y_test[i])


if __name__ == "__main__":
    config_wordalgebra.init()
    main()
