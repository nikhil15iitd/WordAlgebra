import os, datetime
import globals
from dataset import read_draw, numbers_to_words, derivation_to_equation
from template_parser import debug

import numpy as np
import scoring_function as sf

from keras.preprocessing.sequence import pad_sequences
from dsbox_spen.dsbox.spen.utils.metrics import token_level_loss_ar, token_level_loss
'''
from keras.layers import Bidirectional, LSTM, Conv1D, Dense, PReLU, MaxPool1D, Input, Embedding, TimeDistributed, \
    BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from dsbox_spen.dsbox.spen.core import spen as sp, config, energy
from dsbox_spen.dsbox.spen.utils.metrics import token_level_loss_ar, token_level_loss
'''
GLOVE_DIR = 'glove.6B'
EMBEDDING_DIM = 50  # 50
MAX_SEQUENCE_LENGTH = 105

worddict = {}
text2sols = {}

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

def evaluate_citation(xinput=None, yinput=None, yt=None):
    xd = xinput
    yd = yinput
    debug = False
    size = np.shape(xd)[0]
    scorer = sf.Scorer()
    penalty = np.zeros(size)
    for i in range(size):
        x = xd[i, :]
        text = ''
        for j in range(x.shape[0]):
            text += ' ' + worddict[x[j]]
        penalty[i] = scorer.score_output(text, yd[i],text2sols[str(x)])
    return penalty

def main():
    derivations, vocab_dataset = debug()
    X, Y, Z = derivations
    for key in vocab_dataset:
        worddict[vocab_dataset[key]] = key

    print(X[0])
    print(Y[0])
    print(Z[0])

    X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=globals.PROBLEM_LENGTH)
    for i in range(X.shape[0]):
        text2sols[str(X[i])] = Z[i]
    print('='*50)
    #print(text2sols)

    ##################################
    #  Determine components greedily
    ##################################
    vocab_size = len(vocab_dataset.keys())
    SLOT_DIMS = [230]+[MAX_SEQUENCE_LENGTH]*6
    SEARCH_NUM = 50
    DERIVATION_SIZE = 7

    # test with ypred for 1 example
    #ypred = [np.random.randint(globals.PROBLEM_LENGTH) for _ in range(7)]
    ypred = np.random.randint(0, MAX_SEQUENCE_LENGTH, (X.shape[0], DERIVATION_SIZE))
    print(ypred)


    # #(data_sample) * #(slots) * #(possible values for that slot) = 514 * 7 * 105 = 377790 iterations
    start = datetime.datetime.now()
    for slot in range(DERIVATION_SIZE): # determine the value slot by slot
        print('#'*100)
        cur_max = 0
        cur_max_score = -np.inf

        for i in range(X.shape[0]): # for every data sample

            for n in range(SLOT_DIMS[slot]): # try every possible value for that slot
                #candid = np.random.randint(SLOT_DIMS[slot])
                ypred[slot] = n

                #start = datetime.datetime.now()
                #score = evaluate_citation(X[:1], np.array([ypred]*1), Y[:1]) # 5-15ms for 1 example
                score = evaluate_citation(np.array([X[i]]), np.array([ ypred[i] ]), np.array([Y[i]])) # 5-15ms for 1 example
                #end = datetime.datetime.now()
                #print(score)
                #print((end-start).total_seconds()*1000)
                #print(cur_max)
                #print(score)
                #print(cur_max < score)

                # take the maximum
                if cur_max_score < score:
                    cur_max = n
                    cur_max_score = score

        print(cur_max)
        print(cur_max_score)
        #ypred[slot] = cur_max
        ypred[i][slot] = cur_max
    end = datetime.datetime.now()
    print('time took:')
    print((end-start).total_seconds())

    print('#'*100)
    print(len(ypred))
    print(ypred)
    print(Y[0])
    hm_ts, ex_ts = token_level_loss(ypred, Y)
    print(hm_ts)
    print(ex_ts)

    for pred in ypred[:20]:
        print(pred)





if __name__ == "__main__":
    globals.init()  # Fetch global variables such as PROBLEM_LENGTH
    main()
