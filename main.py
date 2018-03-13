import globals
from dataset import read_draw, numbers_to_words, derivation_to_equation
from template_parser import debug
import os
import numpy as np
from keras.layers import Bidirectional, LSTM, Conv1D, Dense, PReLU, MaxPool1D, Input, Embedding, TimeDistributed, \
    BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from dsbox_spen.dsbox.spen.core import spen, config, energy
from dsbox_spen.dsbox.spen.utils.metrics import f1_score, hamming_loss

GLOVE_DIR = 'glove.6B'
EMBEDDING_DIM = 50  # 50
MAX_SEQUENCE_LENGTH = 105


def feed_forward_mlp_model(input_shape, vocab_size, emb_layer):
    '''
    Deep neural network model to get F(x) which is to be fed to SPENs
    '''
    inputs = Input(shape=input_shape)
    emb = emb_layer(inputs)
    l0 = Bidirectional(LSTM(32, return_sequences=True))(emb)
    l0 = Bidirectional(LSTM(32))(l0)
    l1 = Dense(7, activation='softmax', name='t_id')(l0)
    l2 = Dense(vocab_size, activation='softmax', name='m')(l0)
    l3 = Dense(vocab_size, activation='softmax', name='n')(l0)
    l4 = Dense(globals.TEMPLATE_LENGTH, activation='softmax', name='a')(l0)
    l5 = Dense(globals.TEMPLATE_LENGTH, activation='softmax', name='b')(l0)
    l6 = Dense(globals.TEMPLATE_LENGTH, activation='softmax', name='c')(l0)
    l7 = Dense(globals.TEMPLATE_LENGTH, activation='softmax', name='d')(l0)
    l8 = Dense(globals.TEMPLATE_LENGTH, activation='softmax', name='e')(l0)
    l9 = Dense(globals.TEMPLATE_LENGTH, activation='softmax', name='f')(l0)
    model = Model(inputs=inputs, outputs=[l1, l2, l3, l4, l5, l6, l7, l8, l9])
    model.compile('adam', {'t_id': 'sparse_categorical_crossentropy', 'm': 'sparse_categorical_crossentropy',
                           'n': 'sparse_categorical_crossentropy', 'a': 'sparse_categorical_crossentropy',
                           'b': 'sparse_categorical_crossentropy', 'c': 'sparse_categorical_crossentropy',
                           'd': 'sparse_categorical_crossentropy', 'e': 'sparse_categorical_crossentropy',
                           'f': 'sparse_categorical_crossentropy'})
    return model


def get_layers():
    layers = [(1000, 'relu')]
    enlayers = [(250, 'softplus')]
    return (layers, enlayers)


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
                                trainable=False)
    return embedding_layer


def main():
    derivations, vocab_dataset = debug()
    X, Y = derivations
    X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=globals.PROBLEM_LENGTH)
    emb_layer = load_glove(vocab_dataset)
    F = feed_forward_mlp_model(X.shape[1:], len(vocab_dataset.keys()), emb_layer)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=globals.SEED)
    ntrain = X_train.shape[0]
    print(X_train.shape)
    print(y_train.shape)
    F.fit(X_train,
          [y_train[:, 0], y_train[:, 1], y_train[:, 2], y_train[:, 3], y_train[:, 4], y_train[:, 5], y_train[:, 6],
           y_train[:, 7], y_train[:, 8]], batch_size=128, epochs=50, validation_data=(
            X_test, [y_test[:, 0], y_test[:, 1], y_test[:, 2], y_test[:, 3], y_test[:, 4], y_test[:, 5], y_test[:, 6],
                     y_test[:, 7], y_test[:, 8]]))

    # y_pred = np.argmax(F.predict(X_test), axis=2)
    # for i in range(y_pred.shape[0]):
    #     print(derivation_to_equation(y_pred[i].reshape((globals.TEMPLATE_LENGTH,))))
    #     print(derivation_to_equation(y_test[i].reshape((globals.TEMPLATE_LENGTH,))))

    ###Configurable parameters START
    ln = 1e10
    l2 = 0.0
    lr = 0.001
    inf_iter = 10
    inf_rate = 0.1
    mw = 100.0
    dp = 0.0
    bs = 100
    output_num = Y.shape[1]
    input_num = X.shape[1]
    f_layers, en_layers = get_layers()

    config.l2_penalty = l2
    config.inf_iter = inf_iter
    config.inf_rate = inf_rate
    config.learning_rate = lr
    config.dropout = dp
    config.dimension = len(vocab_dataset.keys())
    config.output_num = input_num
    config.input_num = input_num
    config.en_layer_info = en_layers
    config.layer_info = f_layers
    config.margin_weight = mw
    config.lstm_hidden_size = 15
    config.sequence_length = 105
    ###Configurable parameters END


    s = spen.SPEN(config)
    e = energy.EnergyModel(config)
    # s.eval = lambda xd, yd, yt : f1_score_c_ar(yd, yt)
    s.get_energy = e.get_energy_rnn_mlp_emb
    # s.prediction_net = e.softmax_prediction_network
    s.train_batch = s.train_supervised_batch

    s.construct_embedding(EMBEDDING_DIM, len(vocab_dataset.keys()) + 1)
    s.construct(training_type=spen.TrainingType.SSVM)
    s.print_vars()

    s.init()
    s.init_embedding(F.get_layer('embedding_1').get_weights()[0])
    labeled_num = min((ln, ntrain))
    indices = np.arange(labeled_num)
    xlabeled = X_train[indices][:]
    ylabeled = y_train[indices][:]

    total_num = xlabeled.shape[0]
    for i in range(1, 100):
        bs = min((bs, labeled_num))
        perm = np.random.permutation(total_num)

        for b in range(int(ntrain / bs)):
            indices = perm[b * bs:(b + 1) * bs]

            xbatch = xlabeled[indices][:]
            xbatch = np.reshape(xbatch, (xbatch.shape[0], -1))
            ybatch = ylabeled[indices][:]
            ybatch = pad_sequences(ybatch, maxlen=globals.PROBLEM_LENGTH, padding='post', truncating='post', value=0.)
            ybatch = np.reshape(ybatch, (ybatch.shape[0], -1))
            s.set_train_iter(i)
            s.train_batch(xbatch, ybatch)

        if i % 2 == 0:
            yval_out = s.map_predict(xinput=np.reshape(X_test, (X_test.shape[0], -1)))
            ytest_out = pad_sequences(y_test, maxlen=globals.PROBLEM_LENGTH, padding='post', truncating='post',
                                      value=0.)
            ts_f1 = f1_score(yval_out, np.reshape(ytest_out, (ytest_out.shape[0], -1)))
            print(yval_out.shape)
            print(ts_f1)


if __name__ == "__main__":
    globals.init()  # Fetch global variables such as PROBLEM_LENGTH
    config = config.Config()
    main()
