import numpy as np
import sympy
import nltk

vocab_template = {' ': 0, 'm': 1, 'n': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, '0.01': 9, '*': 10, '+': 11,
                  '-': 12, '=': 13, ',': 14, '0': 15, '1': 16, '2': 17, '3': 18, '4': 19, '5': 20}
inv_map = {v: k for k, v in vocab_template.items()}
MAX_TEMPLATE_LENGTH = 30

'''
operators = {'+': 1, '-': 2, '*': 3, '/': 4, '=': 5}
knowns = {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10, 'f': 11, 'g': 12}
unknowns = {'m': 13, 'n': 14, 'l': 15, 'o': 16, 'p': 17, 'q': 18}
symbols = {'%': 19}
separators = {',': 20}
all_template_vars = {0: ' ', 20: ','}
'''
operators = ['+', '-', '*', '/', '=']
coeffs = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
unknowns = ['m', 'n', 'l', 'o', 'p', 'q']
numbers = ['0.01', '0', '1', '2', '3', '4', '5']
seperators = [',', ' ']

opsep = ['+', '-', '*', '/', '=', ',', ' ']
all_symbols = list(vocab_template.keys())
#nonrepeatable = operators+coeffs+

class Scorer(object):
    def __init__(self):
        pass

    def score_output(self, text, ypred, alignment, ytrue):
        """
        ypred: e.g. "a * m + b * n = c"
        """
        score = 0.0
        # words = text.strip().split(' ')
        words = nltk.word_tokenize(text)
        numbers_present = 0
        print(ypred.shape)
        print(ypred)
        ypred_symbol = [ inv_map[int(i)] for i in ypred ]

        #print(ytrue)
        print(ypred_symbol)

        # counting items: https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        unique, counts = np.unique(ypred_symbol, return_counts=True)
        cnt_dict = dict(zip(unique, counts))

        pnlty = 100
        L = ypred.shape[0]
        print(L)


        ###########################################################
        # some hand-crafted rules for valid templates (equations) #
        ###########################################################
        # operators shouldn't be at the start or end
        if ypred_symbol[0] in opsep or ypred_symbol[L-1] in opsep: score -= pnlty
        # there should be at least one '=' symbol
        if not '=' in ypred_symbol: score -= pnlty
        # there shouldn't be more than two '=' symbols or equation seperators
        if '=' in cnt_dict.keys() and cnt_dict['='] > 2:score -= pnlty
        if ',' in cnt_dict.keys() and cnt_dict[','] > 2: score -= pnlty


        # sequential rules
        for i in range(ypred.shape[0]):

            if i > 0 and i <= MAX_TEMPLATE_LENGTH:
                #####################################################
                # Scores ypred and x only
                #####################################################
                # no operators/coeffs/unknowns, etc should be repeated in a row
                # 9C2 combinations?
                if ypred[i-1] in operators and ypred[i] in operators: score -= pnlty
                if ypred[i-1] in coeffs:
                    if ypred[i] in coeffs: score -= pnlty
                    if ypred[i] in unknowns: score -= pnlty
                    if ypred[i] in numbers: score -= pnlty
                if ypred[i-1] in unknowns:
                    if ypred[i] in coeffs: score -= pnlty
                    if ypred[i] in unknowns: score -= pnlty
                    if ypred[i] in numbers: score -= pnlty
                if ypred[i-1] in numbers:
                    if ypred[i] in coeffs: score -= pnlty
                    if ypred[i] in unknowns: score -= pnlty
                    if ypred[i] in numbers: score -= pnlty

                # no operators immediately after an equation separator
                if ypred[i-1] == ',' and ypred[i] in operators: score -= pnlty


            #####################################################
            # Scores from supervised signal (true YSeq or ytrue)
            #####################################################
            if i >= 0 and i <= MAX_TEMPLATE_LENGTH:
                diff = abs(ypred[i] - ytrue[i])
                score -= diff

            if i > MAX_TEMPLATE_LENGTH and ypred[i] != ' ': # trying to make the predictions after max len become pad symbols
                score -= 1000


        return score
