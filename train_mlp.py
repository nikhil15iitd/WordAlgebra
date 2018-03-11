import numpy as np
import json, os

from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import keras
from keras.layers import Bidirectional, LSTM, Conv1D, Dense, PReLU, MaxPool1D, Input, Embedding, TimeDistributed, \
    BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import globals
from dataset import read_draw, numbers_to_words, derivation_to_equation
from template_parser import debug

def load_glove(vocab):
    # ref: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    GLOVE_DIR = 'glove.6B'
    EMBEDDING_DIM = 50#50
    MAX_SEQUENCE_LENGTH = 105
    vocab_size = len(vocab.keys())
    word_index = vocab_size

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.%dd.txt'%EMBEDDING_DIM))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    #embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    embedding_matrix = np.zeros((vocab_size+1, EMBEDDING_DIM))

    #for word, i in word_index.items():
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    from keras.layers import Embedding

    #embedding_layer = Embedding(len(word_index) + 1,
    embedding_layer = Embedding(vocab_size + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer


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


def feed_forward_mlp_model_coeffs(batch_size, input_shape, vocab_size, emb_layer):
    '''
    Deep neural network model to get F(x) which is to be fed to SPENs
    '''
    print(input_shape) # 105,
    print(vocab_size) # => 183
    print(keras.__version__)
    num_output = 7 # since the template vector has 7 elems (without unknowns)

    inputs = Input(shape=input_shape)
    #emb = Embedding(vocab_size, 16, input_length=globals.PROBLEM_LENGTH)(inputs) # => (?, 105, 16)
    emb = emb_layer(inputs)

    #l0 = keras.layers.GRU(32, return_sequences=False)(emb) # => (?, 32)
    print('emb shape:') # (?, 105, 16)
    print(emb.shape)
    l0 = keras.layers.Flatten()(emb)
    l0 = keras.layers.Dense(128, activation='relu')(l0)
    l0 = keras.layers.Dense(128, activation='relu')(l0)

    '''
    # 9 Dense layers for predicting word index in each slot
    outputs = []
    for i in range(num_output):
        if i == 0:
            outputs.append( Dense(230, activation='softmax')(l0) ) # template
        elif i > 0 and i < num_output+1: # i < 7
            outputs.append( Dense(globals.PROBLEM_LENGTH, activation='softmax')(l0) ) # coeffs
    print('output shape:')
    print(outputs[0].shape) # => batch_size x 7
    '''

    #Dense(num_output, activation='linear')(l0)
    l0 = Dense(num_output)(l0) # => outputs logits

    #model = Model(inputs=inputs, outputs=outputs)
    model = Model(inputs=inputs, outputs=l0)

    #optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #model.compile(optimizer, 'sparse_categorical_crossentropy', metrics=['acc'])
    model.compile(optimizer, loss=derivation_loss, metrics=['acc'])

    return model


############################################################################################
#   Custom loss function
#   ref: https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras
############################################################################################
from keras import backend as K
# ref: https://stackoverflow.com/questions/46594115/euclidean-distance-loss-function-for-rnn-keras
def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

def derivation_loss(y_true, y_pred):
    """
    @param: y_true: A tensor containing true labels.
    @param: y_pred:
    """
    #print('debug:')
    #print(y_true.shape) # (?, ?)
    #print(y_pred.shape) # (?, 7)
    return euc_dist_keras(y_true, y_pred)


def custom_loss():
    def derivation(y_true, y_pred):
        return -derivation_loss(y_true, y_pred)
    return derivation


def get_layers():
    layers = [(1000, 'relu')]
    enlayers = [(250, 'softplus')]
    return (layers, enlayers)


def main():
    template_size = 7
    batch_size = 128

    X, Y, vocab_dataset = debug()
    X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=globals.PROBLEM_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=globals.SEED)
    ntrain = X_train.shape[0]
    print(X_train.shape)
    print(X_test.shape)

    vocab_size = len(vocab_dataset.keys())
    emb_layer = load_glove(vocab_dataset)
    F = feed_forward_mlp_model_coeffs(batch_size, X.shape[1:], vocab_size, emb_layer)
    #F.summary()

    '''
    targets = []
    for i in range(template_size):
        targets.append(y_train[:,i])
    test_targets = []
    for i in range(template_size):
        test_targets.append(y_test[:,i])
    '''


    #F.fit(X_train, targets, batch_size=batch_size, epochs=100, validation_data=(X_test, test_targets))
    F.fit(X_train, y_train, batch_size=batch_size, epochs=1000, validation_data=(X_test, y_test))
    F.save('baseline_debug.h5')
    print('='*100)
    print(F.predict(X_test)[0].shape) # 2, 230
    print(F.predict(X_test)[1].shape) # 2, 183
    pred_train = F.predict(X_test)

    print('preds shape:')
    print(pred_train.shape) # (2, 7)
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
    globals.init()
    main()
