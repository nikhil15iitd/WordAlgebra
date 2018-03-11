import globals
from dataset import read_draw, numbers_to_words, derivation_to_equation
from template_parser import debug

import numpy as np
from keras.layers import Bidirectional, LSTM, Conv1D, Dense, PReLU, MaxPool1D, Input, Embedding, TimeDistributed, \
    BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from dsbox_spen.dsbox.spen.core import spen, config, energy
from dsbox_spen.dsbox.spen.utils.metrics import f1_score, hamming_loss


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
    inputs = Input(shape=input_shape)
    l0 = Embedding(vocab_size, 16, input_length=globals.PROBLEM_LENGTH)(inputs)
    l0 = Bidirectional(LSTM(32, return_sequences=True))(l0)
    l0 = Bidirectional(LSTM(32))(l0)
    l0 = Dense(8, activation='softmax')(l0)
    model = Model(inputs=inputs, outputs=l0)
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])
    return model


def get_layers():
    layers = [(1000, 'relu')]
    enlayers = [(250, 'softplus')]
    return (layers, enlayers)


def main():
    X, Y, vocab_dataset = debug()
    print(X)
    X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=globals.PROBLEM_LENGTH)
    print(Y)
    F = feed_forward_mlp_model(X.shape[1:], len(vocab_dataset.keys()))
    # X, Y = read_draw()
    # X, Y = pad_lengths_to_constant(X, Y)
    # F = feed_forward_model(X.shape[1:], len(vocab.keys()), len(all_template_vars.keys()))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=globals.SEED)
    ntrain = X_train.shape[0]
    print(X_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    F.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))
    y_pred = np.argmax(F.predict(X_test), axis=2)
    for i in range(y_pred.shape[0]):
        print(derivation_to_equation(y_pred[i].reshape((TEMPLATE_LENGTH,))))
        print(derivation_to_equation(y_test[i].reshape((TEMPLATE_LENGTH,))))

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
    config.dimension = 21
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

    s.construct_embedding(16, len(vocab.keys()))
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
            ytest_out = pad_sequences(y_test, maxlen=globals.PROBLEM_LENGTH, padding='post', truncating='post', value=0.)
            ts_f1 = f1_score(yval_out, np.reshape(ytest_out, (ytest_out.shape[0], -1)))
            print(yval_out.shape)
            print(ts_f1)


if __name__ == "__main__":
    globals.init() # Fetch global variables such as PROBLEM_LENGTH
    config = config.Config()
    main()
