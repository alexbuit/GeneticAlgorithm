from typing import Callable
from scipy.stats import cauchy
from time import time

import numpy as np
import math
import random as rand


np.random.seed(10)


def t_roulette_sel(tsize=int(1e6), bitsize=4):
    """

    :param tsize: Size of the population
    :param bitsize: Size of a parent within the population

    Correct % => If > 100% or <100% there is are double indexes in the list.
    :return:
    """
    tsart = time()
    rpop = rand_bit_pop(tsize, bitsize)
    # print(rpop)
    parent_list = roulette_select(b2int(rpop))

    tl = []
    for parent in parent_list:
        tl.append(parent[0])
        tl.append(parent[1])

    # print(len(tl), ":", len(set(tl)))
    corrperc = 100 - ((len(tl) - len(set(tl))) / (tsize / 2)) * 100

    t = time() - tsart
    return t, corrperc


def b2int(bit: np.ndarray) -> np.ndarray:
    """
    Conversion of m x n (big endian) bit array to integers.
    :param bit: m x n ndarray of numpy integers (0, 1) representing a bit
    :return: m x n ndarray of integers
    """
    # credits Geoffrey Andersons solution
    # https://stackoverflow.com/questions/41069825/convert-binary-01-numpy-to-integer-or-binary-string
    m, n = bit.shape  # number of columns is needed, not bits.size
    a = 2 ** np.arange(n) # [::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bit @ a  # this matmult is the key line of code

print(b2int(np.array([[0, 0, 1, 1]])))

def b2sfloat(bit: np.ndarray) -> np.ndarray:
    """
    Conversion of m x n (big endian) bit array (numpy) to IEEE 754 single precision float
    according to Jaime's solution
    (https://stackoverflow.com/questions/26018448/convert-a-binary-string-into-ieee-754-single-precision-python)
    :param bit: m x n ndarray of numpy integers representing a bit
    :return: m x n ndarray of IEEE 754 single precision floats
    """
    bit = bit.transpose()
    return np.packbits(bit.reshape(-1, 8)).reshape(-1, 4)[:, ::-1].copy().view(
        np.float32).transpose()[0]


def b2dfloat(bit: np.ndarray) -> np.ndarray:
    """
    Conversion of bit m x n (big endian) bit array (numpy) to IEEE 754 double precision float
    according to Jaime's solution
    (https://stackoverflow.com/questions/26018448/convert-a-binary-string-into-ieee-754-single-precision-python)
    :param bit: m x n ndarray of numpy integers representing a bit
    :return: m x n ndarray of IEEE 754 double precision floats
    """
    bit = bit.transpose()
    return np.packbits(bit.reshape(-1, 16)).reshape(-1, 8)[:, ::-1].copy().view(
        np.float64).transpose()[0]


def rand_bit_pop(n: int, m: int) -> np.ndarray:
    """
    Generate a random bit population
    :param n: Population size dtype int
    :param m: Bitsize dtype int
    :return: List of random bits with a bit being a ndarray array of 0 and 1.
    """
    return np.array([np.random.randint(0, 2, size=m) for _ in range(n)])


def normalrand_bit_pop_float(n, bitsize, upper, lower):

    pop_float = np.linspace(lower, upper, num=n)
    if bitsize == 32:

        return None

    elif bitsize == 64:

        return None

    else:
        pass
    return None


def cauchyrand_bit_pop_float(n, m, loc, scale):
    if m == 32:
        pass

    elif m == 64:
        pass

    return None

print("b2int 32")
print(b2int(rand_bit_pop(10, 32)))
print("b2sfloat 32")
print(b2sfloat(rand_bit_pop(10, 32)))
print("b2int 64")
print(b2int(rand_bit_pop(10, 64)))
print("b2dfloat 64")
print(b2dfloat(rand_bit_pop(10, 64)))

def wheelers_ridge(x1: float, x2: float, a: float = 1.5) -> float:
    """
    Compute the Wheelersridge function for given x1 and x2
    :param x1: First input
    :param x2: Second input
    :param a: additional parameter typically a=1.5
    :return: Value f(x1, x2, a), real float
    """
    return -np.exp(-(x1 * x2 - a) ** 2 - (x2 - a) ** 2)


def michealewicz(x: list, m: float = 10.0) -> float:
    """
    Compute the Micealewicz function for x1, x2, x...
    :param x: List of x inputs, where N-dimensions = len(x)
    :param m: Steepness parameter, typically m=10
    :return: Value f(x1, x2, ....), real float
    """
    return sum(
        [np.sin(x[i]) * np.sin((i * x[i] ** 2) / np.pi) ** (2 * m) for i in
         range(len(x))])


def roulette_select(pop, fx):
    y = fx(b2sfloat(pop))

    y = np.max(y) - y
    yc = y.copy()
    yrng = np.asarray(range(y.size))
    p = y / sum(y)

    pind = []
    for i in range(int(y.size / 2)):
        if p.size > 1:
            try:
                par = np.random.choice(yrng, 2, p=p, replace=False)
            except ValueError:
                p = np.full(p.size, 1 / p.size)
                par = np.random.choice(yrng, 2, p=p, replace=False)

            pind.append(list(sorted(par).__reversed__()))

            yc = np.delete(yc, np.where(yrng == pind[-1][0])[0][0])
            yc = np.delete(yc, np.where(yrng == pind[-1][1])[0][0])
            yrng = np.delete(yrng, np.where(yrng == pind[-1][0])[0][0])
            yrng = np.delete(yrng, np.where(yrng == pind[-1][1])[0][0])
            p = yc / sum(yc)

    return pind


def cross_parents(parent1, parent2):
    a = np.random.rand(0, parent1.size)
    print(parent1.size)

    child = None
    return child


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
    # fx = np.vectorize(fx)
    for _ in range(max_iter):
        # Parents function should return pairs of parent indexes in pop
        parents = select(b2int(pop), fx)
        # Apply the cross function to all parent couples
        children = [cross(pop[p[0]], pop[p[1]]) for p in parents]
        # Mutate the population (without elitism)
        pop = mutate(children)

    return pop


tsart = time()

print("t: ", time() - tsart)
# rpop = rand_bit_pop(10000, 4)
# # # print(rpop)
# roulette_select(b2int(rpop))
# print("t: ", time() - tsart)

# resdata = np.zeros(3)
# step, stop = 100, 10000
# bitsize = 4
#
# for i in np.arange(step, stop + step, step):
#     print("%s / %s" % (i, stop))
#     # rpop = rand_bit_pop(i, bitsize)
#     test = t_roulette_sel(i, bitsize)
#     resdata = np.vstack([resdata, [i, test[0], test[1]]])
#
# np.savetxt("resdata%s_%s_%s.txt" % (bitsize, step, stop), X=resdata, delimiter=";",
#            header="tsize;time;corrperc")
#
# tsize = resdata[:, 0]
# timedata = resdata[:, 1]
# corrdata = resdata[:, 2]
#
# from AdrianPack.Aplot import Default
#
# # def fx(x, a, b):
# #     return a * np.exp(x) + b
#
# graph = Default(tsize, timedata, x_label="Population size", y_label="Time (s)",
#                 degree=2)
# print(graph.fit_coeffs)
# # graph2 = Default(tsize, corrdata, add_mode=True)
# # graph += graph2
# graph()
# # graph.save_as = "resdata%s_%s_%s.png"
# # graph()
#
#

