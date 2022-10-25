from typing import Callable, Union, Iterable

import numpy
from scipy.stats import cauchy
from time import time

import numpy as np
import math
import random as rand
import struct

from helper import *


np.random.seed(10)


bdict = {8: [1, 4, 3], 16: [1, 5, 10], 32: [1, 8, 23], 64: [1, 11, 52],
         128: [1, 15, 112], 256: [1, 19, 236]}

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
    parent_list = roulette_select(b2int(rpop), tfx)

    tl = []
    for parent in parent_list:
        tl.append(parent[0])
        tl.append(parent[1])

    # print(len(tl), ":", len(set(tl)))
    corrperc = 100 - ((len(tl) - len(set(tl))) / (tsize / 2)) * 100

    t = time() - tsart
    return t, corrperc


def rand_bit_pop(n: int, m: int) -> np.ndarray:
    """
    Generate a random bit population
    :param n: Population size dtype int
    :param m: Bitsize dtype int
    :return: List of random bits with a bit being a ndarray array of 0 and 1.
    """
    return np.array([np.random.randint(0, 2, size=m) for _ in range(n)])


def normalrand_bit_pop_float(n, bitsize, lower, upper):

    pop_float = np.linspace(lower, upper, num=n)
    blist = []
    if bitsize == 32:
        for val in range(pop_float.size):
            blist.append(floatToBinary32(pop_float[val]))
            # tval, tres = pop_float[val], b2sfloat(floatToBinary64(pop_float[val]))[0]
            # try: np.testing.assert_almost_equal(tres, tval )
            # except AssertionError: print("Fail")

    elif bitsize == 64:
        for val in range(pop_float.size):
            blist.append(floatToBinary64(pop_float[val]))
            # tval, tres = pop_float[val], b2dfloat(blist[-1])[0]
            # try: np.testing.assert_almost_equal(tres, tval )
            # except AssertionError: print("Fail")

    else:
        pass
    return np.array(blist)


def cauchyrand_bit_pop_float(shape: Union[Iterable, float], bitsize: int, loc: float,
                             scale: float) -> np.ndarray:
    if isinstance(shape, int):
        shape = (shape, 1)
    elif len(shape) == 1:
        shape = (shape[0], 1)

    size = shape[0] * shape[1]

    pop_float = cauchy.rvs(loc=loc, scale=scale, size=size)
    pop_float = np.array(np.array_split(pop_float, int(size/shape[0])), dtype=float)
    blist = []
    for val in range(pop_float.shape[0]):
        blist.append(float2Ndbit(pop_float[:, val], bitsize))

    return np.array(blist)


def tfx(x):
    return 3 * x**2 + 2 * x + 1


def wheelers_ridge(x: list, a: float = 1.5) -> float:
    """
    Compute the Wheelersridge function for given x1 and x2
    :param x: list with x1 (otype: float) and x2 (otype: float)
    :param a: additional parameter typically a=1.5
    :return: Value f(x1, x2, a), real float
    """
    x1, x2 = x
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

    y = np.zeros(pop.shape)
    for val in range(pop.shape[0]):
        pass

    y = fx(b2dfloat(pop))
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


def cross_parents64(parent1, parent2):
    # parent1, parent2 = np.flip(parent1), np.flip(parent2)
    child = np.zeros(parent1.size, dtype=np.uint8)
    # print(parent1, parent2)
    # print(child)
    p1sign = parent1[0]
    p2sign = parent2[0]

    if np.random.randint(0, 1):
        child[0] = p2sign
    else:
        child[0] = p1sign


    p1exp = parent1[1:12]
    p2exp = parent2[1:12]
    #
    # for i in range(1, 11):
    #     if np.random.randint(0, 1):
    #         child[i] = p2exp[i]
    #     else:
    #         child[i] = p1exp[i]
    child[1:12] = parent1[1:12]

    c1 = np.random.randint(12, parent1.size-1)
    c2 = np.random.randint(c1 + 1, parent1.size)

    child[12:c1] = parent1[12:c1]
    child[c1:c2] = parent2[c1:c2]
    child[c2:] = parent1[c2:]

    child_float = b2dfloat(child.transpose())

    return child


def cross_parents(parent1, parent2, bitsize):
    global bdict

    child = np.zeros(parent1.shape, dtype=np.uint8)

    nbits = int(parent1.size/bitsize)

    p1_1 = parent1[:nbits * (1 + bdict[bitsize][1])]
    p2_1 = parent2[:nbits * (1 + bdict[bitsize][1])]

    if np.random.randint(0, 1):
        child[:nbits * (1 + bdict[bitsize][1])] = p1_1[:nbits * (1 + bdict[bitsize][1])]
    else:
        child[:nbits * (1 + bdict[bitsize][1])] = p2_1[:nbits * (1 + bdict[bitsize][1])]

    c1 = np.random.randint(nbits * (1 + bdict[bitsize][1]), parent1.size - 1)
    c2 = np.random.randint(c1 + 1, parent1.size)

    child[nbits * (1 + bdict[bitsize][1]):c1] = parent1[nbits * (1 + bdict[bitsize][1]):c1]
    child[c1:c2] = parent2[c1:c2]
    child[c2:] = parent1[c2:]

    return child


def mutate(bit, bitsize, **kwargs):
    global bdict

    bitc = bit.copy()

    mutate_coeff = 1/bitsize
    if "mutate_coeff" in kwargs:
        mutate_coeff = kwargs["mutate_coeff"]

    mutations = np.random.randint(bdict[bitsize][2], bit.size, int(1/mutate_coeff))

    # Speed up?
    for mutation in mutations:
        if bitc[mutation]:
            bitc[mutation] = 0
        else:
            bitc[mutation] = 1

    return bitc


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

if __name__ == "__main__":
    tsart = time()


    size = (1000, 100)
    genlist = []


    # rpop = normalrand_bit_pop_float(10000, 64, -5, 5)
    rpop = cauchyrand_bit_pop_float(size, 64, 0, 5)
    print(rpop.shape)
    parents = roulette_select(rpop, tfx)

    epochs = int(np.floor(np.log2(size)))

    for j in range(epochs):
        print("%s/%s" % (j+1, epochs))
        newgen = []
        for ppair in parents:
            newgen.append(cross_parents64(rpop[ppair[0]], rpop[ppair[1]]))

        genlist.append(b2dfloat(rpop))
        rpop = np.array(newgen)
        print(rpop.shape)
        parents = roulette_select(np.array(newgen), tfx)
    genlist.append(b2dfloat(rpop[0]))
    genlist = genlist
    genarr = np.full((len(genlist), genlist[0].size), np.NAN)

    k = 0
    for i in genlist:
        for j in range(i.size):
            genarr[k, j] = i[j]
        k += 1
    genarr = genarr.transpose()
    print(genarr)
    print(b2dfloat(rpop[0]))

    dataind = 0
    np.savetxt("tdataGAmult_%s.txt" % dataind, genarr, delimiter=";",
               header="".join("%s;" %i for i in range(len(genlist) + 1)))
    # from AdrianPack.Aplot import LivePlot
    #
    # print(genlist)
    #
    # def livefunc(i):
    #     print(i)
    #     return tfx(genlist[i])
    #
    # LP = LivePlot(x=genlist, x_label="x data", y_label="y data")
    # LP.run(interval=100)


    print("t: ", time() - tsart)



