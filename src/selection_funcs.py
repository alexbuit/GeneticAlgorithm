
import numpy as np
from helper import *
from typing import Iterable

def calc_fx(pop, fx, bitsize, nbit2num = Ndbit2float, **kwargs):
    return np.apply_along_axis(fx, 1, nbit2num(pop, bitsize, **kwargs))


def lin_fitness(y, a, b):
    return a * y + b


def exp_fitness(y, k):
    return y ** k


def sigmoid_fitness(x, k, x0 = 0):
    return 1 / (1 + np.exp(-k * (x - x0)))


def sort_list(y, p):
    """
    Select parents out the pool pop represented by their  (in same order as pop in ga) from their
    corresponding
    :param y: np.ndarray with values f(x1, x2, x3, x...) in order of pop
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
    """
    pop, fx, bitsize, nbit2num = Ndbit2float, **kwargs

    :param args:
        args, kwargs passed onto the calc_fx function
    :param kwargs:
        k => parameteres passed onto the fitness function
        fitness_func => the function to determine the fitness of a value y
        calculated from input population
    :return:
    """
    y = calc_fx(*args, **kwargs)
    y = np.max(y) - y

    k = [1.5]
    if "k" in kwargs:
        k = kwargs["k"]

    if not isinstance(k, Iterable):
        k = [k]

    fitness_func = exp_fitness
    if "fitness_func" in kwargs:
        fitness_func = kwargs["fitness_func"]

    fitness = fitness_func(y, *k)

    p = fitness / sum(fitness)

    return sort_list(fitness, p)



def fitness_method(*args, **kwargs):
    """

    :param args:
        args, kwargs passed onto the calc_fx function
    :param kwargs:
        k => parameteres passed onto the fitness function
        fitness_func => the function to determine the fitness of a value y
        calculated from input population
    :return:
    """
    y = calc_fx(*args, **kwargs)
    y = np.max(y) - y

    k = [1.5]
    if "k" in kwargs:
        k = kwargs["k"]

    if not isinstance(k, Iterable):
        k = [k]

    fitness_func = exp_fitness
    if "fitness_func" in kwargs:
        fitness_func = kwargs["fitness_func"]

    fitness = fitness_func(y, *k)
    return fitness


def rank_selection(*args, **kwargs):
    """

    calc_fx params:
    pop, fx, bitsize, nbit2num = Ndbit2float, **kwargs

    :param args:
    :param kwargs:
    :return:
    """
    y = calc_fx(*args, **kwargs)
    y = np.max(y) - y

    # probability paramter for rank selection
    prob_param = 1.9
    if "p" in kwargs:
        prob_param = kwargs["p"]

    # parameters for fitness func
    k = [1.5]
    if "k" in kwargs:
        k = kwargs["k"]

    if not isinstance(k, Iterable):
        k = [k]

    # fitness func
    fitness_func = exp_fitness
    if "fitness_func" in kwargs:
        fitness_func = kwargs["fitness_func"]

    fitness = fitness_func(y, *k)
    fit_rng = np.argsort(fitness)

    p = (prob_param * (1 - prob_param)**(np.arange(1, fitness.size + 1, dtype=float) - 1))
    p = p/np.sum(p)

    selection_array = np.zeros(fit_rng.shape)
    for i in range(fitness.size):
        selection_array[fit_rng[i]] = p[i]

    pind = []
    rng = np.arange(0, y.size)

    for i in range(int(y.size / 2)):
        if selection_array.size > 1:
            try:
                par = np.random.choice(rng, 2, p=selection_array, replace=False)
            except ValueError:
                print("Value error in selection, equal probability")
                selection_array = np.full(selection_array.size, 1 / selection_array.size)
                par = np.random.choice(rng, 2, p=selection_array, replace=False)

            pind.append(list(sorted(par).__reversed__()))

            fitness[pind[-1][0]] = -1
            fitness[pind[-1][1]] = -1

        fit_rng = np.argsort(fitness)

        p = (prob_param * (1 - prob_param) ** (
                    np.arange(1, (fitness.size) + 1, dtype=float) - 1))
        p = p / np.sum(p)

        for i in range(fitness.size):
            selection_array[fit_rng[i]] = p[i]

        for i in range(fitness.size):
            if fitness[i] == -1:
                selection_array[i] = 0

    return pind

if __name__ == "__main__":
    from population_initatilisation import *
    from t_functions import *

    pop = bit8([15, 4], 16)
    pop_float = ndbit2int(pop, 16, factor=5)

    tst_fx = michealewicz

    parents = rank_selection(pop=pop, fx=tst_fx, bitsize=16, nbit2num=ndbit2int,
                   factor=5, p=0.1, k=0.01)
    print(parents)


