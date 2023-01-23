import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable

try:
    from .helper import *
    # from .gradient_descent import gradient_descent
except ImportError:
    from dfmcontrol.helper import *
    # from dfmcontrol.gradient_descent import gradient_descent

def calc_fx(pop, fx, bitsize, nbit2num = Ndbit2floatIEEE754, **kwargs):
    """ Calculate the fitness of a population.
    :param pop: Population of individuals
    :param fx: Function to calculate the fitness of an individual
    :param bitsize: Bitsize of an individual
    :param nbit2num: Function to convert a ndarray of bits to a number
    :param kwargs: Additional arguments for nbit2num
    :return: Fitness of the population
    """
    if "bitsize" in kwargs["b2nkwargs"]:
        kwargs["b2nkwargs"].pop("bitsize")

    return np.apply_along_axis(fx, 1, nbit2num(pop, bitsize, **kwargs["b2nkwargs"]))


def lin_fitness(y, a, b):
    """
    Scale fitness linearly

    :param y: np.ndarray with values f(x1, x2, x3, x...) in order of pop
    :param a: parameter a
    :param b: parameter b
    :return: np.ndarray with linearised values f(x1, x2, x3, x...) in order of pop
    """
    return a * y + b


def exp_fitness(y, k):
    """
    Scale fitness exponentially
    :param y: np.ndarray with values f(x1, x2, x3, x...) in order of pop
    :param k: parameter k
    :return: np.ndarray with exponential values f(x1, x2, x3, x...) in order of pop
    """
    y = y/np.max(y)
    return 1 / (k ** y)

def simple_fitness(y, *args):
    """
     Normalise fitness
    :param y: np.ndarray with fitness values f(x1, x2, x3, x...) in order of pop
    :param args: Pipeline for additional arguments that are included by default (IE a, b or k)
    :return: np.ndarray with normalised values f(x1, x2, x3, x...) in order of pop
    """
    y = np.abs(y)
    return np.max(y) - y


def sigmoid_fitness(x, k, x0 = 0):
    """
    Sigmoid function
    :param x: np.ndarray with values f(x1, x2, x3, x...) in order of pop
    :param k:  steepness parameter k
    :param x0: start pos parameter x0
    :return: np.ndarray with sigmoid values f(x1, x2, x3, x...) in order of pop
    """
    return 1 / (1 + np.exp(-k * (x - x0)))


def probability_sort_list(y, p, allow_duplicates=False, **kwargs):
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

            if not allow_duplicates:
                y = np.delete(y, np.where(rng == pind[-1][0])[0][0])
                y = np.delete(y, np.where(rng == pind[-1][1])[0][0])
                rng = np.delete(rng, np.where(rng == pind[-1][0])[0][0])
                rng = np.delete(rng, np.where(rng == pind[-1][1])[0][0])
                p = y / sum(y)

    return pind

def sort_list(y, p, **kwargs):
    """
    Select parents out the pool pop
    :param y: np.ndarray with values f(x1, x2, x3, x...) in order of pop
    :param p: np.ndarray with probablity values correlating to y values
    :param kwargs:  allow_duplicates => allow duplicates in the selection
    :return:  selected parents based on their probability
    """

    pind = []

    for i in range(int(y.size / 2)):
        par = np.flip(np.argsort(p))[0:2]
        par = list(sorted(par).__reversed__())
        pind.append(par)
        try:
            y = np.delete(y, np.where(par == pind[-1][0])[0][0])
            y = np.delete(y, np.where(par == pind[-1][1])[0][0])
        except IndexError:
            pass

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
    return sort_list(fitness, p, **kwargs), fitness, p, y

def tournament_selection(*args, **kwargs):
    """
    pop, fx, bitsize, nbit2num = Ndbit2float, **kwargs

    :param args:
    :param kwargs:
    :return:
    """
    y = calc_fx(*args, **kwargs)



def rank_selection(*args, **kwargs):
    """
    Select parents out the pool pop based on their rank in the fitness function

    calc_fx params: pop, fx, bitsize, nbit2num = Ndbit2float, **kwargs

    :param args: args, kwargs passed onto the calc_fx function
    :param kwargs: k => parameteres passed onto the fitness function
    :return: selected parents based on their probability
    """
    y = calc_fx(*args, **kwargs)

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

    fitness = fitness_func(y, np.asarray(k))

    fit_rng = np.argsort(fitness)

    p = np.abs((prob_param * (1 - prob_param)**(np.arange(1, fitness.size + 1, dtype=float) - 1)))
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
                if kwargs["verbosity"] == 1:
                    print("Value error in selection, equal probability")

                selection_array = np.full(selection_array.size, 1 / selection_array.size)
                par = np.random.choice(rng, 2, p=selection_array, replace=False)

            pind.append(list(sorted(par).__reversed__()))

    return pind, fitness, p, y


def rank_space_selection(*args, **kwargs):
    """
        Select parents out the pool pop according to rank space selection

        calc_fx params: pop, fx, bitsize, nbit2num = Ndbit2float, **kwargs
        :param args: args, kwargs passed onto the calc_fx function
        :param kwargs: k => parameteres passed onto the fitness function
        :return: selected parents based on their probability
        """
    pop = args[0]
    y = calc_fx(*args, **kwargs)

    # probability paramter for rank selection
    prob_param = kwargs.get("p", 1.9)

    # diversity parameter for significance of the distance between individuals
    div_param = kwargs.get("d", 1)

    # parameters for fitness func
    k = kwargs.get("k", [1.5])

    # gradient descent parameter, if 0 then no gradient descent
    gd_param = kwargs.get("gd", 0.1)

    if not isinstance(k, Iterable):
        k = [k]

    # fitness func
    fitness_func = exp_fitness
    if "fitness_func" in kwargs:
        fitness_func = kwargs["fitness_func"]

    fitness = fitness_func(y, *k)
    fit_rng = np.argsort(fitness)

    best = fit_rng[0]
    diversity = np.sqrt(np.asarray([pop[best]**2 - pop[i]**2 for i in pop])) * div_param

    if gd_param > 0:
        # gradient descent
        pass

    fitness = fitness + diversity
    fit_rng = np.argsort(fitness)

    p = (prob_param * (1 - prob_param) ** (
                np.arange(1, fitness.size + 1, dtype=float) - 1))
    p = p / np.sum(p)


def boltzmann_selection(*args, **kwargs):
    pass



if __name__ == "__main__":
    from pop import *
    from test_functions import *

    pop = bitpop([16, 4], 16)
    pop_float = ndbit2int(pop, 16, factor=5)

    tst_fx = michealewicz

    print(michealewicz([0, 0]))
    print(wheelers_ridge([0, 0]))
    print(ackley([0, 0]))


    parents = rank_space_selection(pop=pop, fx=tst_fx, bitsize=16, nbit2num=ndbit2int,
                   factor=5, p=0.1, k=0.01)
    print(parents)


