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
SEPERATOR = ',' # equation separator symbol
PAD = ' '       # padding symbol

opsep = ['+', '-', '*', '/', '=', ',']
opseppad = ['+', '-', '*', '/', '=', ',', ' ']

all_symbols = list(vocab_template.keys())
all_valid_symbols = operators+coeffs+unknowns+numbers+[SEPERATOR]
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
        #print(ypred.shape)
        #print(ypred)
        #print(ytrue)
        ypred_symbol = [ inv_map[int(i)] for i in ypred ]
        ytrue_symbol = [ inv_map[int(i)] for i in ytrue ]
        #print(ypred_symbol)

        # counting items: https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        unique, counts = np.unique(ypred_symbol, return_counts=True)
        cnt_dict = dict(zip(unique, counts))

        pnlty = 10 # 1
        strong_pnlty = 20 # 1000
        reward = 1
        strong_reward = 2
        L = ypred.shape[0]


        ###########################################################
        # some hand-crafted rules for valid templates (equations) #
        ###########################################################
        # operators shouldn't be at the start or end
        if ypred_symbol[0] in opseppad or ypred_symbol[L-1] in opsep: score -= strong_pnlty # higher penalty because it'd be extremely invalid
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
                if ypred_symbol[i-1] in operators:
                    if ypred_symbol[i] in opseppad: score -= pnlty
                if ypred_symbol[i-1] in coeffs:
                    if ypred_symbol[i] in coeffs: score -= pnlty
                    if ypred_symbol[i] in unknowns: score -= pnlty
                    if ypred_symbol[i] in numbers: score -= pnlty
                if ypred_symbol[i-1] in unknowns:
                    if ypred_symbol[i] in coeffs: score -= pnlty
                    if ypred_symbol[i] in unknowns: score -= pnlty
                    if ypred_symbol[i] in numbers: score -= pnlty
                if ypred_symbol[i-1] in numbers:
                    if ypred_symbol[i] in coeffs: score -= pnlty
                    if ypred_symbol[i] in unknowns: score -= pnlty
                    if ypred_symbol[i] in numbers: score -= pnlty

                # more specific rules
                # there shouldn't be any '0 + ...' or '0 * ...'
                if ypred_symbol[i-1] == '0' and ypred_symbol[i] in operators: score -= pnlty

                # reward consecutive pad symbols a bit so that the model wants to put a series of pad in the end?
                if ypred_symbol[i-1] == PAD and ypred_symbol[i] == PAD: score += reward
                if ypred_symbol[i-1] == PAD and ypred_symbol[i] in all_valid_symbols: score -= strong_pnlty

                # no operators immediately after an equation separator
                if ypred_symbol[i-1] == ',' and ypred_symbol[i] in operators: score -= pnlty


            #####################################################
            # Scores from supervised signal (true YSeq or ytrue)
            #####################################################
            if i >= 0 and i <= MAX_TEMPLATE_LENGTH: # need to compute diff in integers for this one so use ypred and ytrue
                diff = abs(ypred[i] - ytrue[i])
                score -= diff

            if i > MAX_TEMPLATE_LENGTH and ypred_symbol[i] != ' ': # trying to make the predictions after max len become pad symbols
                score -= strong_pnlty


        return score
