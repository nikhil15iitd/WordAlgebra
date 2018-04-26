import os, datetime
import globals
from word2number import w2n
from dataset import read_draw, numbers_to_words, derivation_to_equation
from template_parser import debug

import numpy as np
import scoring_function as sf

from keras.preprocessing.sequence import pad_sequences
from dsbox_spen.dsbox.spen.utils.metrics import token_level_loss_ar, token_level_loss

GLOVE_DIR = 'glove.6B'
EMBEDDING_DIM = 50  # 50
MAX_SEQUENCE_LENGTH = 105

worddict = {}
text2sols = {}
text2align = {}


def evaluate_citation(xinput=None, yinput=None, yt=None):
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
        penalty[i] = scorer.score_output(text, yd[i], text2sols[str(x)], text2align[str(x)])
    return penalty


def is_number(s):
    try:
        float(s.replace(',', ''))
        return True
    except ValueError:
        return False


def main():
    derivations, vocab_dataset = debug()
    X, Xtags, YSeq, Y, Z = derivations
    for key in vocab_dataset:
        worddict[vocab_dataset[key]] = key

    X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=globals.PROBLEM_LENGTH)
    for i in range(X.shape[0]):
        text2sols[str(X[i])] = Z[i]
        text2align[str(X[i])] = Y[i]
    print('=' * 50)

    ##################################
    #  Determine components greedily
    ##################################
    vocab_size = len(vocab_dataset.keys())
    SLOT_DIMS = [25] + [MAX_SEQUENCE_LENGTH] * 6
    SEARCH_NUM = 50
    DERIVATION_SIZE = 7

    # test with ypred for 1 example
    # ypred = [np.random.randint(globals.PROBLEM_LENGTH) for _ in range(7)]
    ypred = np.random.randint(0, MAX_SEQUENCE_LENGTH, (X.shape[0], DERIVATION_SIZE))
    print(ypred)
    print ("SLOTDIM" + str(SLOT_DIMS))

    # #(data_sample) * #(slots) * #(possible values for that slot) = 514 * 7 * 105 = 377790 iterations
    start = datetime.datetime.now()
    for slot in range(DERIVATION_SIZE):  # determine the value slot by slot
        print('#' * 100)
        cur_max = 0
        cur_max_score = -np.inf

        for i in range(X.shape[0]):  # for every data sample
            possible=[]
            if slot==0:
                for k in range(1,SLOT_DIMS[slot]+1):
                    possible.append(k)
            else:
                for index in range(0,len(X[i])):
                    if str.isnumeric(worddict[X[i][index]])==True:
                        possible.append(index)
                    else:
                        try:
                            if(worddict[X[i][index]].find('-'))==-1 and worddict[X[i][index]]!="point" :
                                t = w2n.word_to_num(worddict[X[i][index]])
                                #print(worddict[X[i][index]])
                                possible.append(t)
                        except:
                            flag=1
            for n in possible:  # try every possible value for that slot
                # candid = np.random.randint(SLOT_DIMS[slot])

                ypred[slot] = n
                #print(np.array(X[i][0]))
                # start = datetime.datetime.now()
                score = evaluate_citation(np.array([X[i]]), np.array([ypred[i]]),
                                          np.array([Y[i]]))  # 5-15ms for 1 example?
                # end = datetime.datetime.now()
                # print(score)
                # print((end-start).total_seconds()*1000)

                # take the maximum value for prediction
                if cur_max_score < score:
                    cur_max = n
                    cur_max_score = score
            # print(cur_max)
            # print(cur_max_score)
            ypred[i][slot] = cur_max
    end = datetime.datetime.now()
    print('time took:')
    print((end - start).total_seconds())  # => 3148.295232 = about an hour

    print('#' * 100)
    print(len(ypred))
    print(ypred)
    print(Y[0])
    hm_ts, ex_ts = token_level_loss(ypred, Y)
    print(hm_ts)  # => 0.9756809338521402
    print(ex_ts)  # => 0.02431906614785991

    # Outputs the first 20 predictions for debugging

    for pred in ypred[:20]:
        print(pred)
    '''
    =>
    [229 229 229 229 229 229 229]
    [104 104 104 104 104 104 104]
    [104 104 104 104 104 104 104]
    [104 104 104 104 104 104 104]
    [104 104 104 104 104 104 104]
    [104 104 104 104 104 104 104]
    [104 104 104 104 104 104 104]
    [68 86 16  5  1 10 17]
    [102  65   1  84   5  52  93]
    [92 79 95 64 14 95 30]
    [35 17 71 41 97 82 40]
    [ 1  2 39 70 33 41 97]
    [ 22  52  65 102  57  87  68]
    [44 35 69  8 37 94 20]
    [82 53 15 99 80 59 41]
    [ 9  6 44 10 75 12 33]
    [ 85  52 101  70  12 103  54]
    [71 53 16 49 88 44 18]
    [95  8 84  8 46 91 58]
    [ 70  77  50   2  23  15 103]
    '''


if __name__ == "__main__":
    globals.init()  # Fetch global variables such as PROBLEM_LENGTH
    main()
