import numpy as np
import sympy


class Scorer(object):
    def __init__(self):
        self.coefficients = 0

    def template_one(self, a, b, c, d, e, f):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, b, c), (d, e, f)))
        return sympy.solve_linear_system(system, m, n)

    def template_two(self, a, b, c, d):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((0.01 * a, 0.01 * b, c), (1, 1, d)))
        return sympy.solve_linear_system(system, m, n)

    def template_three(self, a, b, c, d):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, b, c), (1, 1, d)))
        return sympy.solve_linear_system(system, m, n)

    def template_four(self, a, b, c):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, a, b), (c, -c, b)))
        return sympy.solve_linear_system(system, m, n)

    def template_five(self, a, b, c):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((a, -b, 0), (1, 1, c)))
        return sympy.solve_linear_system(system, m, n)

    def template_six(self, a, b, c):
        m = sympy.symbols('x')
        return sympy.solve(a * m + b * m - c, m)

    def template_seven(self, a, b):
        m = sympy.symbols('x')
        n = sympy.symbols('y')
        system = sympy.Matrix(((1, 1, a), (-b, 1, 0)))
        return sympy.solve_linear_system(system, m, n)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def score_output(self, text, ypred, sols):
        score = 0.0
        words = text.split(' ')
        for i in range(ypred.shape[0]):
            if i > 2 and ypred[i] > 0:
                score -= 1
        for word in words:
            if self.is_number(word):
                score += 1

        # only solve when each coefficient has a corresponding number in text
        if score == 0:
            if ypred[0] == 0:
                solutions = self.template_one(ypred[3], ypred[4], ypred[5], ypred[6], ypred[7], ypred[8])
                score -= abs(solutions['x'] - sols[0]) + abs(solutions['y'] - sols[1])
            elif ypred[0] == 1:
                solutions = self.template_two(ypred[3], ypred[4], ypred[5], ypred[6])
                score -= abs(solutions['x'] - sols[0]) + abs(solutions['y'] - sols[1])
            elif ypred[0] == 2:
                solutions = self.template_three(ypred[3], ypred[4], ypred[5], ypred[6])
                score -= abs(solutions['x'] - sols[0]) + abs(solutions['y'] - sols[1])
            elif ypred[0] == 3:
                solutions = self.template_four(ypred[3], ypred[4], ypred[5])
                score -= abs(solutions['x'] - sols[0]) + abs(solutions['y'] - sols[1])
            elif ypred[0] == 4:
                solutions = self.template_five(ypred[3], ypred[4], ypred[5])
                score -= abs(solutions['x'] - sols[0]) + abs(solutions['y'] - sols[1])
            elif ypred[0] == 5:
                solutions = self.template_six(ypred[3], ypred[4], ypred[5])
                score -= abs(solutions['x'] - sols[0])
            elif ypred[0] == 6:
                solutions = self.template_seven(ypred[3], ypred[4])
                score -= abs(solutions['x'] - sols[0]) + abs(solutions['y'] - sols[1])

        return score
