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
        return {'x': sympy.solve(a * m + b * m - c, m)}

    def template_7(self, a, b):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((1, 1, a), (-b, 1, 0)))
        return sympy.solve_linear_system(system, m, n)

    def template_8(self, a, b, c, d):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, b, c*d), (1, 1, c)))
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
        return {'x': sympy.solve(a*m - b, m)}

    def template_12(self, a, b):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((1, 1, a), (1, -1, b)))
        return sympy.solve_linear_system(system, m, n)

    def template_13(self, a, b, c, d):
        m = sympy.symbols('x')
        return {'x' : sympy.solve(a*m - b*m - c + d, m)}

    def template_14(self, a, b, c):
        m = sympy.symbols('x')
        return {'x' : sympy.solve(0.01*a*m - b + c, m)}

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
        return {'x' : sympy.solve(0.01*a*m - b + c, m)}

    def template_17(self, a, b, c, d):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, -b, c), (1, 1, d)))
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
        return ret, -diff

    def score_output(self, text, ypred, sols):
        score = 0.0
        words = text.split(' ')
        numbers_present = 0
        for i in range(ypred.shape[0]):
            if i > 0:
                if self.is_number(words[ypred[i]]):
                    numbers_present += 1
                    score += 1
                else:
                    score -= 1
            if i == 0 and ypred[i] >= 6:
                score -= 20000

        if len(sols) > 0:
            try:
                # print (self.find_closest_num(words, ypred[1]))

                a, pnlty = float(words[self.find_closest_num(words, ypred[1])])
                score += pnlty
                b, pnlty = float(words[self.find_closest_num(words, ypred[2])])
                score += pnlty
                c, pnlty = float(words[self.find_closest_num(words, ypred[3])])
                score += pnlty
                d, pnlty = float(words[self.find_closest_num(words, ypred[4])])
                score += pnlty
                e, pnlty = float(words[self.find_closest_num(words, ypred[5])])
                score += pnlty
                f, pnlty = float(words[self.find_closest_num(words, ypred[6])])
                score += pnlty

                if ypred[0] == 0 and numbers_present >= 6:
                    solutions = self.template_one(a, b, c, d, e, f)
                    print(solutions)
                    score -= 10 * abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1])
                elif ypred[0] == 1 and numbers_present >= 4:
                    solutions = self.template_two(a, b, c, d)
                    print(solutions)
                    score -= 10 * abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1])
                elif ypred[0] == 2 and numbers_present >= 4:
                    solutions = self.template_three(a, b, c, d)
                    print(solutions)
                    score -= 10 * abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1])
                elif ypred[0] == 3 and numbers_present >= 3:
                    solutions = self.template_four(a, b, c)
                    print(solutions)
                    score -= 10 * abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1])
                elif ypred[0] == 4 and numbers_present >= 3:
                    solutions = self.template_five(a, b, c)
                    print(solutions)
                    score -= 10 * abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1])
                elif ypred[0] == 5 and numbers_present >= 3:
                    solutions = self.template_six(a, b, c)
                    print(solutions)
                    score -= 10 * abs(solutions["x"] - sols[0])
                elif ypred[0] == 6 and numbers_present >= 2:
                    solutions = self.template_seven(a, b)
                    print(solutions)
                    score -= 10 * abs(solutions["x"] - sols[0]) + abs(solutions['y'] - sols[1])
            except:
                print(words)
                # solve template

        return score
