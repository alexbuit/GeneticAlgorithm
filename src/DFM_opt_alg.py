
from typing import Callable
from scipy.stats import cauchy
from time import time

import numpy as np
import math
import random as rand


def b2int(bit):
    # credits Geoffrey Andersons solution
    # https://stackoverflow.com/questions/41069825/convert-binary-01-numpy-to-integer-or-binary-string
    m, n = bit.shape  # number of columns is needed, not bits.size
    a = 2 ** np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bit @ a  # this matmult is the key line of code

def rand_bit_pop(n: int, m: int) -> np.ndarray:
    """
    Generate a random bit population
    :param n: Population size dtype int
    :param m: Bitsize dtype int
    :return: List of random bits with a bit being a ndarray array of 0 and 1.
    """
    return np.array([np.random.randint(0, 2, size=m) for _ in range(n)])


def wheelers_ridge(x1: float, x2: float, a: float = 1.5) -> float:
    """
    Compute the Wheelersridge function for given x1 and x2
    :param x1: First input
    :param x2: Second input
    :param a: additional parameter typically a=1.5
    :return: Value f(x1, x2, a), real float
    """
    return -np.exp(-(x1 * x2 - a)**2 - (x2 - a)**2)


def michealewicz(x: list, m: float = 10.0) -> float:
    """
    Compute the Micealewicz function for x1, x2, x...
    :param x: List of x inputs, where N-dimensions = len(x)
    :param m: Steepness parameter, typically m=10
    :return: Value f(x1, x2, ....), real float
    """
    return sum([np.sin(x[i]) * np.sin((i * x[i]**2)/np.pi)**(2*m) for i in
                range(len(x))])


def roulette_select(y):
    y = np.max(y) - y
    p = y/np.max(y)
    print(p)

    # print(len([np.where(y == i)[0][0] for i in ys]))
    # print(len(list(dict.fromkeys([np.where(y == i)[0][0] for i in ys]))))
    return None


def geneticalg(fx: Callable, pop: np.ndarray, max_iter: int, select: Callable,
               cross: Callable, mutate: Callable):
    """

    :param fx:

    :param pop:
    :param max_iter:
    :param select:
    :param cross:
    :param mutate:
    :return:
    """

    for _ in range(max_iter):
        # Parents function should return pairs of parent indexes in pop
        parents = select(pop, fx)
        # Apply the cross function to all parent couples
        children = [cross(pop[p[0]], pop[p[1]]) for p in parents]
        # Mutate the population (without elitism)
        pop = mutate(children)

    return pop

tsart = time()
rpop = rand_bit_pop(1000, 16)
# print(rpop)
roulette_select(b2int(rpop))
print("t: ", time() - tsart)

