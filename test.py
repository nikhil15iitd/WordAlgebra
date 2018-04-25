import scoring_function as sf
from template_parser import debug
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import globals


scorer = sf.Scorer()
derivations, vocab_dataset = debug()
X, Xtags, Y, Z = derivations
worddict = {}
text2sols = {}
text2align = {}

for key in vocab_dataset:
    worddict[vocab_dataset[key]] = key
X = pad_sequences(X, padding='post', truncating='post', value=0., maxlen=105)
for i in range(X.shape[0]):
    text2sols[str(X[i])] = Z[i]
    text2align[str(X[i])] = Y[i]
xd = X
yd = Y
size = np.shape(xd)[0]
scorer = sf.Scorer()
penalty = np.zeros(size)
count  = 0;
for i in range(size):
    x = xd[i, :]
    text = ''
    for j in range(x.shape[0]):
        text += ' ' + worddict[x[j]]
    penalty[i] = scorer.score_output(text, yd[i], text2sols[str(x)], text2align[str(x)])
    if(penalty[i]<-1):
        count+=1
print ("done")



