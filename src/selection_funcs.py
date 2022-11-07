
import numpy as np
from helper import *

def calc_fx(pop, fx, bitsize, nbit2num = Ndbit2float, **kwargs):
    return np.apply_along_axis(fx, 1, nbit2num(pop, bitsize, **kwargs))


def lin_fitness(y, a, b):
    return a * y + b


def exp_fitness(y, k):
    return y ** k


def sort_list(y, p):
    """

    :param y: np.ndarray with values f(x1, x2, x3, x...)
    :param p: np.ndarray with probablity values correlating to y values
              so that p[1] has contains the probablity of y[1]
    :return: selected parents based on their probability
    """
    pind = []
    rng = np.arange(0, y.size)

    for i in range(int(y.size / 2)):
        if p.size > 1:
            try:
                par = np.random.choice(rng, 2, p=p, replace=False)
            except ValueError:
                p = np.full(p.size, 1 / p.size)
                par = np.random.choice(rng, 2, p=p, replace=False)

            pind.append(list(sorted(par).__reversed__()))

            y = np.delete(y, np.where(rng == pind[-1][0])[0][0])
            y = np.delete(y, np.where(rng == pind[-1][1])[0][0])
            rng = np.delete(rng, np.where(rng == pind[-1][0])[0][0])
            rng = np.delete(rng, np.where(rng == pind[-1][1])[0][0])
            p = y / sum(y)

    return pind

# def roulette_select(pop, fx, bitsize, nbit2num = Ndbit2float):
#
#     y = np.apply_along_axis(fx, 1, nbit2num(pop, bitsize))
#     y = np.max(y) - y
#     p = y / sum(y)
#
#     return sort_list(y, p)

def roulette_selection(*args, **kwargs):
    y = calc_fx(*args, **kwargs)
    y = np.max(y) - y

    k = 1.5
    if "k" in kwargs:
        k = kwargs["k"]

    fitness = exp_fitness(y, k)
    p = fitness / sum(fitness)

    return sort_list(y, p)

def fitness_method(*args, **kwargs):
    y = calc_fx(*args, **kwargs)
    y = np.max(y) - y

    k = 1.5
    if "k" in kwargs:
        k = kwargs["k"]

    fitness = exp_fitness(y, k)
    return fitness


