import globals
from dataset import read_draw, numbers_to_words, derivation_to_equation
from template_parser import debug
import os
import nltk
import numpy as np
import scoring_function_template as sf

from keras.layers import Bidirectional, LSTM, Flatten, Dense, PReLU, MaxPool1D, Input, Embedding, TimeDistributed, \
    BatchNormalization, concatenate, Conv1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from dsbox_spen.dsbox.spen.core import spen as sp, config, energy
from dsbox_spen.dsbox.spen.utils.metrics import token_level_loss_ar, token_level_loss

GLOVE_DIR = 'glove.6B'
EMBEDDING_DIM = 50  # 50
MAX_SEQUENCE_LENGTH = 105

worddict = {}
text2sols = {}
text2align = {}
text2YSeq = {}
vocab_template = {' ': 0, 'm': 1, 'n': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, '0.01': 9, '*': 10, '+': 11,
                  '-': 12, '=': 13, ',': 14, '0': 15, '1': 16, '2': 17, '3': 18, '4': 19, '5': 20}
inv_map = {v: k for k, v in vocab_template.items()}

def feed_forward_mlp_model(input_shape, b_vocab, emb_layer):
    '''
    Deep neural network model to get F(x) which is to be fed to SPENs
    '''

    '''
    enc_inputs = Input(shape=input_shape)
    emb = emb_layer(enc_inputs)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_inputs)
    enc_states = [state_h, state_c]

    dec_inputs = Input(shape=(None, num_dec_tokens))
    dec_outputs, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(dec_inputs, initial_state=enc_states)
    dec_outputs = Dense(num_decoder_tokens, activation='softmax')(decoder_outputs)

    model = Model(inputs=[enc_inputs, dec_inputs], outputs=[dec_outputs])
    '''
    inputs = Input(shape=input_shape)
    emb = emb_layer(inputs)
    l0 = Bidirectional(LSTM(32, return_sequences=True))(emb)
    l0 = Bidirectional(LSTM(32, return_sequences=True))(l0)
    l0 = TimeDistributed(Dense(21, activation='softmax'))(l0) # 21: vocab size for template
    model = Model(inputs=inputs, outputs=l0)
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
                                trainable=True)
    return embedding_layer


def evaluation_function(xinput=None, yinput=None, yt=None):
    xd = xinput
    yd = yinput
    size = np.shape(xd)[0]
    scorer = sf.Scorer()
    penalty = np.zeros(size)
    for i in range(size):
        x = xd[i, :]
        text = ''
        for j in range(x.shape[0]):
            text += ' ' + worddict[x[j]]
        #penalty[i] = scorer.score_output(text, yd[i], text2sols[str(x)], text2align[str(x)])
        #penalty[i] = scorer.score_output(text, yd[i], yt[i])
        penalty[i] = scorer.score_output(text, yd[i], text2align[str(x)], text2YSeq[str(x)])
    return penalty


def main():
    derivations, vocab_dataset = debug()
    X, Xtags, YSeq, Y, Z = derivations
    print(YSeq[0])

    for key in vocab_dataset:
        worddict[vocab_dataset[key]] = key
    X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=globals.PROBLEM_LENGTH)
    Xtags = pad_sequences(Xtags, padding='post', truncating='post', value=0., maxlen=globals.PROBLEM_LENGTH)
    #YSeq = pad_sequences(YSeq, padding='post', truncating='post', value=0., maxlen=30) # max length for templates is 30?
    YSeq = pad_sequences(YSeq, padding='post', truncating='post', value=0., maxlen=globals.PROBLEM_LENGTH) # max length for templates is 30?
    YSeq = np.reshape(YSeq, (YSeq.shape[0], YSeq.shape[1], 1))
    for i in range(X.shape[0]):
        text2sols[str(X[i])] = Z[i]
        text2align[str(X[i])] = Y[i]
        text2YSeq[str(X[i])] = YSeq[i]
    emb_layer = load_glove(vocab_dataset)
    F = feed_forward_mlp_model(X.shape[1:], 36, emb_layer)
    X_train, X_test, Y_train, Y_test, Xtags_train, Xtags_test, YSeq_train, YSeq_test = train_test_split(X, Y, Xtags,
                                                                                                        YSeq,
                                                                                                        test_size=0.2,
                                                                                                        random_state=23)
    ntrain = X_train.shape[0]
    print(X_train.shape)
    print(YSeq.shape)
    #F.fit([Xtrain], [YSeq_train], batch_size=128, epochs=20, validation_data=([X_test], [YSeq_test]))
    F.fit(X_train, YSeq_train, batch_size=128, epochs=10, validation_data=(X_test, YSeq_test))


    y_pred = np.argmax(F.predict(X_test), axis=2)
    y_true = np.squeeze(YSeq_test, axis=2)
    print(y_pred.shape)
    print(YSeq_test.shape)
    print(y_true.shape)

    ypred_symbol = [ inv_map[i] for i in y_pred[0] ]
    print(y_pred[0])
    #print(y_true[0])
    print(ypred_symbol)

    YSeq_train = np.squeeze(YSeq_train, axis=2)
    YSeq_test = np.squeeze(YSeq_test, axis=2)

    # display the actual predictions of the model:
    # ref: https://stackoverflow.com/questions/25345770/list-comprehension-replace-for-loop-in-2d-matrix?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    ypred_symbols = np.array([ [ inv_map[int(j)] for j in row ] for row in y_pred ])
    ytrue_symbols = np.array([ [ inv_map[int(j)] for j in row ] for row in YSeq_test[:, :30] ])
    print(ypred_symbols.shape)
    print(ytrue_symbols.shape)
    print(ypred_symbols)
    print(ytrue_symbols)


    # TODO: fix this part
    global_acc = 0.0
    for i in range(y_pred.shape[0]):
        acc = 0.0
        for j in range(y_pred.shape[1]):
            if y_pred[i][j] == YSeq_test[i][j] and j < 20:
                acc += 1
        global_acc += (acc / y_pred.shape[1])
    print(global_acc)

    ###Configurable parameters START
    ln = 1e10
    l2 = 0.0
    lr = 0.001
    inf_iter = 20
    inf_rate = 0.1
    mw = 1000
    dp = 0.0
    bs = 100
    ip = 0.0
    output_num = 30#YSeq.shape[1]
    input_num = X.shape[1]
    f_layers, en_layers = get_layers()

    config.l2_penalty = l2
    config.inf_iter = inf_iter
    config.inf_rate = inf_rate
    config.learning_rate = lr
    config.dropout = dp
    config.dimension = 21#30+1#globals.PROBLEM_LENGTH + 1
    config.output_num = output_num
    config.input_num = input_num
    config.en_layer_info = en_layers
    config.layer_info = f_layers
    config.margin_weight = mw
    config.lstm_hidden_size = 16
    config.sequence_length = 30#105
    config.inf_penalty = ip
    ###Configurable parameters END
    s = sp.SPEN(config)
    e = energy.EnergyModel(config)
    s.get_energy = e.get_energy_rnn_mlp_emb
    s.evaluate = evaluation_function
    s.train_batch = s.train_unsupervised_batch

    s.createOptimizer()
    s.construct_embedding(EMBEDDING_DIM, len(vocab_dataset.keys()) + 1)
    s.construct(training_type=sp.TrainingType.Rank_Based)
    s.print_vars()

    s.init()
    s.init_embedding(F.get_layer('embedding_1').get_weights()[0])
    labeled_num = min((ln, ntrain))
    indices = np.arange(labeled_num)
    xlabeled = X_train[indices][:]
    ylabeled = YSeq_train[indices][:, :30]#YSeq_train[indices][:]
    xtags_labeled = Xtags_train[indices][:]
    print('Training spen...')
    print(xlabeled.shape)
    print(ylabeled.shape)

    total_num = xlabeled.shape[0]
    NUM_EPOCHS = 100
    for i in range(1, NUM_EPOCHS):
        bs = min((bs, labeled_num))
        perm = np.random.permutation(total_num)

        for b in range(int(ntrain / bs)):
            indices = perm[b * bs:(b + 1) * bs]

            xbatch = xlabeled[indices][:]
            xbatch = np.reshape(xbatch, (xbatch.shape[0], -1))
            ybatch = ylabeled[indices][:]
            ybatch = np.reshape(ybatch, (ybatch.shape[0], -1))
            print('ybatch')
            print(ybatch.shape)
            xtags_batch = xtags_labeled[indices][:]
            xtags_batch = np.reshape(xtags_batch, (xtags_batch.shape[0], -1))
            s.set_train_iter(i)
            s.train_batch(xbatch, verbose=4)

        if i % 2 == 0:
            print('='*100)
            print('%dTH EPOCH'%i)

            # make prediction
            y_pred = s.map_predict(xinput=np.reshape(Xtags_test, (Xtags_test.shape[0], -1)))

            # display predictions and ground truths in string
            np.set_printoptions(threshold=np.nan) # ref: https://stackoverflow.com/questions/1987694/how-to-print-the-full-numpy-array
            ypred_symbols = np.array([ [ inv_map[int(j)] for j in row ] for row in y_pred ])
            ytrue_symbols = np.array([ [ inv_map[int(j)] for j in row ] for row in YSeq_test[:, :30] ])
            print(ypred_symbols)
            #print(ytrue_symbols)
            np.set_printoptions(threshold=1000) # default printing setting

            # compute err and acc
            hm_ts, ex_ts = token_level_loss(y_pred, YSeq_test[:, :30])
            print(hm_ts)
            print(ex_ts)

        print('Trained spen for %d epochs.'%NUM_EPOCHS)

if __name__ == "__main__":
    globals.init()  # Fetch global variables such as PROBLEM_LENGTH
    config = config.Config()
    main()
