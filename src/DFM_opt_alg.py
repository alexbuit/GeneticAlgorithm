from typing import Callable, Union, Iterable

import numpy
from scipy.stats import cauchy
from time import time

import numpy as np
import math
import random as rand
import struct

from population_initatilisation import *
from selection_funcs import *
from t_functions import *


# np.random.seed(10)


bdict = {8: [1, 4, 3], 16: [1, 5, 10], 32: [1, 8, 23], 64: [1, 11, 52],
         128: [1, 15, 112], 256: [1, 19, 236]}



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


class genetic_algoritm:

    def __init__(self, dtype: str = "float", bitsize: int = 32,
                 endianness: str = "big"):
        self.dtype = dtype
        self.bitsize = bitsize
        self.endianness = endianness

    def run(self, size: Iterable = (1000, 1), ):
        pass

    def get_results(self):
        pass

    def save_results(self, path: str):
        pass

    def load_results(self, path: str):
        pass

    def plot_results(self):
        pass


if __name__ == "__main__":
    tsart = time()


    size = (1000, 2)
    low, high = 10, 20
    bitsize = 64
    tfunc = wheelers_ridge

    # epochs = int(np.floor(np.log2(size[0])))
    epochs = 100

    for sim in range(1):
        genlist = []
    # rpop = normalrand_bit_pop_float(10000, 64, -5, 5)
        rpop = uniform_bit_pop_float(size, bitsize, low, high)
        parents = roulette_select(rpop, tfunc, bitsize)

        for j in range(epochs):
            print("%s/%s" % (j+1, epochs))

            newgen = cauchyrand_bit_pop_float([int(size[0]/2), size[1]], bitsize, low, high).tolist()
            for ppair in parents:
                newgen.append(cross_parents(rpop[ppair[0]], rpop[ppair[1]], bitsize))

            # Select top10
            t10 = roulette_select(rpop, tfunc, 64)[:5]
            genlist.append([])
            for ppair in t10:
                genlist[j].append(rpop[ppair[0]])
                genlist[j].append(rpop[ppair[1]])
            genlist[j] = np.array(genlist[j])

            # genlist.append(rpop)
            rpop = np.array(newgen)
            parents = roulette_select(np.array(newgen), tfunc, bitsize)

        # genlist.append(rpop)
        genarr = np.empty((size[0], epochs), dtype=object)

        k = 0
        for i in genlist:
            for j in range(i.shape[0] - 1):

                strbit = "".join(str(s) for s in i[j])
                # print("~~~~~~")
                # print("strbit: %s" % strbit)
                # print("i: %s, k: %s" % (j, k))
                # print(genarr[j, k])
                genarr[j, k] = strbit
                # print(genarr[j, k])
            k += 1

        genarr = genarr.T
        # print(genarr)
        # print(Ndbit2float(rpop[0], bitsize))

        dataind = 5
        with open("GAmult_%s_tfunc%s_bsize%s_sim%s.txt" % (dataind, tfunc.__name__, bitsize, sim), 'w') as f:
            f.write(';'.join([str(i) for i in range(genarr.shape[0])]) + "\n")
            genarr = genarr.T
            for i in range(genarr.shape[0]):
                f.write(";".join([str(item) for item in genarr[i]]) + "\n")

    # np.savetxt("tdataGAmult_%s.txt" % dataind, genarr, delimiter=";",
    #            header="".join("%s;" %i for i in range(len(genlist) + 1)))

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



