from typing import Callable

import numpy
from scipy.stats import cauchy
from time import time

import numpy as np
import math
import random as rand
import struct


np.random.seed(12)


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


def b2sfloat(bit: np.ndarray) -> np.ndarray:
    """
    Conversion of m x n (big endian) bit array (numpy) to IEEE 754 single precision float
    according to Jaime's solution
    (https://stackoverflow.com/questions/26018448/convert-a-binary-string-into-ieee-754-single-precision-python)
    :param bit: m x n ndarray of numpy integers representing a bit
    :return: m x n ndarray of IEEE 754 single precision floats
    """
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
    return np.packbits(bit.reshape(-1, 16)).reshape(-1, 8)[:, ::-1].copy().view(
        np.float64).transpose()[0]


def floatToBinary64(val):
    """
    https://www.technical-recipes.com/2012/converting-between-binary-and-decimal-representations-of-ieee-754-floating-point-numbers-in-c/
    :param value:
    :return:
    """
    value = struct.unpack("Q", struct.pack('d', val))[0]
    if val < 0:
        return np.array([int(i) for i in format(value, "#065b")[2:]])

    if value > 0:
        return np.array([int(i) for i in "0" + format(value, "#065b")[2:]])

    else:
        return np.array([int(i) for i in "".join("0" for _ in range(64))])


def floatToBinary32(val):
    """
    https://www.technical-recipes.com/2012/converting-between-binary-and-decimal-representations-of-ieee-754-floating-point-numbers-in-c/
    :param value:
    :return:
    """
    value = struct.unpack("L", struct.pack('f', val))[0]
    if val < 0:
        return np.array([int(i) for i in format(value, "#033b")[2:]])

    if value > 0:
        return np.array([int(i) for i in "0" + format(value, "#033b")[2:]])

    else:
        return np.array([int(i) for i in "".join("0" for _ in range(32))])


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


def cauchyrand_bit_pop_float(n, bitsize, loc, scale):
    pop_float = cauchy.rvs(loc=loc, scale=scale, size=n)
    blist = []
    if bitsize == 32:
        for val in range(pop_float.size):
            blist.append(floatToBinary32(pop_float[val]))
            # print(pop_float[val], floatToBinary32(pop_float[val]).size)
            # tval, tres = pop_float[val], b2sfloat(floatToBinary32(pop_float[val]))[0]
            # try: np.testing.assert_almost_equal(tres, tval ,decimal=4)
            # except AssertionError:
            #     print(floatToBinary32(pop_float[val]))
            #     print(tres, tval)
            #     print("Fail")

    elif bitsize == 64:
        for val in range(pop_float.size):
            blist.append(floatToBinary64(pop_float[val]))
            # tval, tres = pop_float[val], b2dfloat(floatToBinary64(pop_float[val]))[0]
            # try: np.testing.assert_almost_equal(tres, tval ,decimal=8)
            # except AssertionError:
            #     print(floatToBinary32(pop_float[val]))
            #     print(tres, tval)
            #     print("Fail")

    return np.array(blist)


# print("b2int 32")
# print(b2int(rand_bit_pop(10, 32)))
# print("b2sfloat 32")
# print(b2sfloat(rand_bit_pop(10, 32)))
# print("b2int 64")
# print(b2int(rand_bit_pop(10, 64)))
# print("b2dfloat 64")
# print(b2dfloat(rand_bit_pop(10, 64)))

def tfx(x):
    return 3 * x**2 + 2 * x + 1

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

    # print("p1")
    # print(parent1)
    # print(b2dfloat(parent1))
    # print("p2")
    # print(parent2)
    # print(b2dfloat(parent2))
    # print("Child")
    # print(child)
    # print(b2dfloat(child))

    child_float = b2dfloat(child.transpose())

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

if __name__ == "__main__":
    tsart = time()


    size = 1000
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

    dataind = 6
    np.savetxt("tdataGA_%s.txt" % dataind, genarr, delimiter=";",
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

