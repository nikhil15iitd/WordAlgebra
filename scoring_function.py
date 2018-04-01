import numpy as np
import sympy


class Scorer(object):
    def __init__(self):
        self.coefficients = 0

    def template_1(self, a, b, c, d, e, f):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, b, c), (d, e, f)))
        return sympy.solve_linear_system(system, m, n)

    def template_2(self, a, b, c, d):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((0.01 * a, 0.01 * b, c), (1, 1, d)))
        return sympy.solve_linear_system(system, m, n)

    def template_3(self, a, b, c, d):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, b, c), (1, 1, d)))
        return sympy.solve_linear_system(system, m, n)

    def template_4(self, a, b, c):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, a, b), (c, -c, b)))
        return sympy.solve_linear_system(system, m, n)

    def template_5(self, a, b, c):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, -b, 0), (1, 1, c)))
        return sympy.solve_linear_system(system, m, n)

    def template_6(self, a, b, c):
        m = sympy.symbols('x')
        solved = sympy.solve(a * m + b * m - c, m)
        # print(solved)
        return {'x': solved[0]}

    def template_7(self, a, b):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((1, 1, a), (-b, 1, 0)))
        return sympy.solve_linear_system(system, m, n)

    def template_8(self, a, b, c, d):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, b, c * d), (1, 1, c)))
        return sympy.solve_linear_system(system, m, n)

    def template_9(self, a, b, c, d):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, b, c), (1, -1, d)))
        return sympy.solve_linear_system(system, m, n)

    def template_10(self, a, b, c):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, -1, b), (1, 1, c)))
        return sympy.solve_linear_system(system, m, n)

    def template_11(self, a, b):
        m = sympy.symbols('x')
        solved = sympy.solve(a * m - b, m)
        # print(solved)
        return {'x': solved[0]}

    def template_12(self, a, b):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((1, 1, a), (1, -1, b)))
        return sympy.solve_linear_system(system, m, n)

    def template_13(self, a, b, c, d):
        m = sympy.symbols('x')
        solved = sympy.solve(a * m - b * m - c + d, m)
        # print(solved)
        return {'x': solved[0]}

    def template_14(self, a, b, c):
        m = sympy.symbols('x')
        solved = sympy.solve(0.01 * a * m - b + c, m)
        # print(solved)
        return {'x': solved[0]}

    def template_15(self, a, b, c, d):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, -b, c), (1, 1, d)))
        return sympy.solve_linear_system(system, m, n)

    def template_16(self, a, b, c):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, a, b), (1, -1, c)))
        return sympy.solve_linear_system(system, m, n)

    def template_17(self, a, b, c):
        m = sympy.symbols('x')
        solved = sympy.solve(a * m - b * m - c, m)
        # print(solved)
        return {'x': solved[0]}

    def template_18(self, a, b, c):
        m = sympy.symbols('x')
        solved = sympy.solve(a * m - b + c, m)
        # print(solved)
        return {'x': solved[0]}

    def template_19(self, a, b, c, d):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, b, c), (d, -1, 0)))
        return sympy.solve_linear_system(system, m, n)

    def template_20(self, a, b):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((1, 1, a), (2, 4, b)))
        return sympy.solve_linear_system(system, m, n)

    def template_21(self, a, b, c):
        m = sympy.symbols('x')
        solved = sympy.solve(a * m - b - c, m)
        # print(solved)
        return {'x': solved[0]}

    def template_22(self, a, b, c):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((1, -a, b), (1, 1, c)))
        return sympy.solve_linear_system(system, m, n)

    def template_23(self, a, b, c):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, 0, b), (c, -1, 0)))
        return sympy.solve_linear_system(system, m, n)

    def template_24(self, a, b):
        m = sympy.symbols('x')
        solved = sympy.solve(m - a + b, m)
        # print(solved)
        return {'x': solved[0]}

    def template_25(self, a, b, c):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, a, b), (1, -1, c)))
        return sympy.solve_linear_system(system, m, n)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def find_closest_num(self, words, index):
        ret = -1
        diff = 10000
        for i in range(len(words)):
            if (self.is_number(words[i])):
                if (abs(i - index) < diff):
                    ret = i
                    diff = abs(i - index)
        return ret

    def score_output(self, text, ypred, sols):
        score = 0.0
        words = text.split(' ')
        numbers_present = 0
        for i in range(ypred.shape[0]):
            if i > 0 and ypred[i] >= len(words):
                score -= 100
            if i == 0 and ypred[i] >= 25:
                score -= 100

        if len(sols) > 0:
            try:
                pnlty = abs(ypred[1] - self.find_closest_num(words, ypred[1]))
                a = float(words[self.find_closest_num(words, ypred[1])])
                score -= pnlty
                pnlty = abs(ypred[2] - self.find_closest_num(words, ypred[2]))
                b = float(words[self.find_closest_num(words, ypred[2])])
                score -= pnlty
                pnlty = abs(ypred[3] - self.find_closest_num(words, ypred[3]))
                c = float(words[self.find_closest_num(words, ypred[3])])
                score -= pnlty
                pnlty = abs(ypred[4] - self.find_closest_num(words, ypred[4]))
                d = float(words[self.find_closest_num(words, ypred[4])])
                score -= pnlty
                pnlty = abs(ypred[5] - self.find_closest_num(words, ypred[5]))
                e = float(words[self.find_closest_num(words, ypred[5])])
                score -= pnlty
                pnlty = abs(ypred[6] - self.find_closest_num(words, ypred[6]))
                f = float(words[self.find_closest_num(words, ypred[6])])
                score -= pnlty

                if ypred[0] == 0:
                    solutions = self.template_1(a, b, c, d, e, f)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 1:
                    solutions = self.template_2(a, b, c, d)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 2:
                    solutions = self.template_3(a, b, c, d)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 3:
                    solutions = self.template_4(a, b, c)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 4:
                    solutions = self.template_5(a, b, c)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 5:
                    solutions = self.template_6(a, b, c)
                    score -= (abs(solutions["x"] - sols[0]))
                elif ypred[0] == 6:
                    solutions = self.template_7(a, b)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 7:
                    solutions = self.template_8(a, b, c, d)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 8:
                    solutions = self.template_9(a, b, c, d)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 9:
                    solutions = self.template_10(a, b, c)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 10:
                    solutions = self.template_11(a, b)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]))
                elif ypred[0] == 11:
                    solutions = self.template_12(a, b)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 12:
                    solutions = self.template_13(a, b, c, d)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]))
                elif ypred[0] == 13:
                    solutions = self.template_14(a, b, c)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]))
                elif ypred[0] == 14:
                    solutions = self.template_15(a, b, c, d)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 15:
                    solutions = self.template_16(a, b, c)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 16:
                    solutions = self.template_17(a, b, c)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]))
                elif ypred[0] == 17:
                    solutions = self.template_18(a, b, c)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]))
                elif ypred[0] == 18:
                    solutions = self.template_19(a, b, c, d)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 19:
                    solutions = self.template_20(a, b)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 20:
                    solutions = self.template_21(a, b, c)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]))
                elif ypred[0] == 21:
                    solutions = self.template_22(a, b, c)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 22:
                    solutions = self.template_23(a, b, c)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
                elif ypred[0] == 23:
                    solutions = self.template_24(a, b)
                    score -= (abs(solutions["x"] - sols[0]))
                elif ypred[0] == 24:
                    solutions = self.template_25(a, b, c)
                    # print(solutions)
                    score -= (abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1]))
            except:
                return score
                # print('boo')
                # solve template

        return score
