from typing import Callable, Union, Iterable, Optional

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
        self.bitsize: int = bitsize
        self.endianness: str = endianness

        self.genlist: list = []
        self.pop: np.ndarray = np.array([])

        self.tfunc: Callable = self.none
        self.targs: dict = {}

        self.select: Callable = roulette_select
        self.cross: Callable = cross_parents
        self.mutation: Callable = mutate

        self.seed: Callable = self.none

        self.epochs = None

        self.results: list = []

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def __repr__(self):
        return ""

    def __getitem__(self, item):
        return self.results[item]

    def run(self, selargs: dict = {}, seedargs: dict = {},
            cargs: dict = {},
            epochs: int = 100, verbosity: int = 1):
        """

        :param seedargs:
        :param selargs:
        :param epochs:
        :param verbosity:
        :return:
        """

        if len(self.pop) == 0:
            self.init_pop()

        self.epochs = epochs

        selargs["fx"] = tfunc
        selargs["bitsize"] = bitsize
        parents = self.select(self.pop, **selargs)

        if self.seed.__name__ == "none":
            self.epochs = int(np.floor(np.log2(size[0])))

        for epoch in range(self.epochs):
            if verbosity:
                print("%s/%s" % (epoch + 1, epochs))

            if self.seed.__name__ == "none":
                newgen = []
            else:
                newgen = self.seed(**seedargs).tolist()

            cargs["bitsize"] = self.bitsize
            for ppair in parents:
                newgen.append(self.cross(self.pop[ppair[0]], self.pop[ppair[1]],
                                         **cargs))

            # Select top10
            t10 = self.select(self.pop, self.tfunc, 64)[:]
            self.genlist.append([])
            for ppair in t10:
                self.genlist[epoch].append(self.pop[ppair[0]])
                self.genlist[epoch].append(self.pop[ppair[1]])

            self.genlist[epoch] = np.array(self.genlist[epoch])

            # genlist.append(rpop)
            self.pop = np.array(newgen)
            parents = self.select(np.array(newgen), **selargs)

    def run_threaded(self, threads, **kwargs):

        pass

    def init_pop(self, method: Union[str, Callable] = "uniform", **kwargs):
        """
        set self.pop to an array generated by predefined routines or usermade method.
        :param method:
            Optional[str, callable]
            if str method will be initialised by routines included in population_initialisation.py
            the str should match the init method, so uniform -> 'uniform' and
            cauchy 'cauchy' etc
            full list of usable args:
            ['uniform', 'cauchy']

            if callable the population will be the return value of given function.
            A population function should return a mx1 numpy array of bits, to convert
            floating point values to approved bit values use the float2Ndbit function
            included in helper.py

        :param kwargs:
            kwargs for init method

        :return: None
        """
        if method == "uniform":
            self.pop = uniform_bit_pop_float(**kwargs)
        elif method == "cauchy":
            self.pop = cauchyrand_bit_pop_float(**kwargs)
        else:
            self.pop = method(**kwargs)

        return None

    def set_pop(self, pop: np.ndarray):
        """
        Set population (self.pop) to provided ndarray of bits.
        :param pop:
        np.ndarray of shape mx1, with bits in numpy arrays of dtype: np.uint8
        like:
        [[0, 1, ... ,0, 1], [0, 1, ... ,0, 1], [0, 1, ... ,0, 1]]
        :return: None
        """
        self.pop = pop
        return None

    def get_pop(self):
        """
        Return a copy of population (self.pop)
        :return: self.pop.copy()
        """
        return self.pop.copy()

    def target_func(self, target, targs: dict = None):
        self.tfunc = target
        self.targs = targs

        return None

    def get_results(self):
        return self.genlist

    def set_cross(self, cross: Callable):
        """
        Set the cross method used in the GA, method should take 2 arguments:
        parent1 and parent2 (both np.ndarray of dim 1 with dtype np.uint8)
          + optional kwargs
        and return a single numpy array with binary value of the resulting child
        from p1 and p2.

        :param cross:
            Method to cross binary data p1 and p2 to form child.

        Example method:

        def cross(p1, p2, bitsize, **kwargs):
            # Take the first half of p1 and cross it with the other half of p2
            return pnp.concatenate([p1[:int(1/2 * bitsize)], p2[int(1/2 * bitsize):]])

        It is recommended to add **kwargs to the provided method to be able to
        handle excess arguments.
        :return: None
        """
        self.cross = cross
        return None

    def set_select(self, select: Callable):
        """
        Set the parent selection method used in GA, method should take an
        np.ndarray of dim mx1 as an argument + kwargs and return a list of lists with
        indexes of (unique) combinations.

        :param select:
            Method to select parent combination by returning their indexes in the
            population self.pop.

        Example method:

        def select(pop, **kwargs):
            # Return completely random combinations in pop array
            return np.random.choice(range(pop.shape[0]), 2, replace=False)

        It is recommended to add **kwargs to the provided method to be able to
        handle excess arguments.
        :return: None
        """
        self.select = select
        return None

    def set_mutate(self, mutation: Callable):
        """
        Set the mutation method which takes a single np.ndarray of dim 1 with
        dtype np.uint8 and kwargs to return the mutated bit. The shape of the input
        array should equal the shape of the output array.

        :param mutation:
            Method to mutate a single np.ndarray of bits

        Example method:

        def mutate(bit, **kwargs):
            # select a random point in the bit
            ind = np.random.randint(0, bit.size)
            # if 1 turn 0 else turn 1.
            if bit[ind]:
                bit[ind] = 0
            else:
                 bit[ind] = 1
            # return the mutated bit
            return bit

        It is recommended to add **kwargs to the provided method to be able to
        handle excess arguments.
        :return:
        """
        self.mutation = mutation
        return None

    def save_results(self, path: str):
        genarr = np.empty((self.genlist[0].shape[0] - 1, self.epochs), dtype=object)
        k = 0
        for i in self.genlist:
            for j in range(i.shape[0] - 1):
                strbit = "".join(str(s) for s in i[j])
                genarr[j, k] = strbit
            k += 1

        with open(path, 'w') as f:
            f.write(';'.join([str(i) for i in range(genarr.shape[1])]) + "\n")
            for i in range(genarr.shape[0]):
                f.write(";".join([str(item) for item in genarr[i]]) + "\n")

    def load_results(self, path: str):
        from AdrianPack.Fileread import Fileread

        data = list(Fileread(path=path)().values())

        for i in range(len(data)):
            data[i] = np.delete(data[i], np.where(data[i] == "None"))

            resmat = np.empty((len(data[i]), len(data[i][0])), dtype=np.uint8)
            for x in range(len(data[i])):
                for b in range(len(data[i][x])):
                    resmat[x, b] = int(data[i][x][b])

            data[i] = resmat

        self.results = [Ndbit2float(i, 64) for i in data]

        return self.results

    def plot_results(self):
        pass

    @staticmethod
    def none(*args, **kwargs):
        return None


if __name__ == "__main__":
    tsart = time()


    size = (10000, 2)
    low, high = 0, 100
    bitsize = 64
    tfunc = wheelers_ridge

    # epochs = int(np.floor(np.log2(size[0])))
    epochs = 100

    ga = genetic_algoritm(bitsize=64)
    ga.init_pop("uniform", shape=(1000, 2), low=0, high=100, bitsize=64)
    ga.seed = uniform_bit_pop_float
    ga.target_func(tfunc)
    ga.run(seedargs={"shape": (int(size[0]/2), 2), "bitsize": 64, "low": low,
                     "high": high}, epochs=10)
    ga.save_results("test.txt")

    # for sim in range(1):
    #     genlist = []
    # # rpop = normalrand_bit_pop_float(10000, 64, -5, 5)
    #     rpop = uniform_bit_pop_float(size, bitsize, low, high)
    #     parents = roulette_select(rpop, tfunc, bitsize)
    #
    #     for j in range(epochs):
    #         print("%s/%s" % (j+1, epochs))
    #
    #         newgen = uniform_bit_pop_float([int(size[0]/2), size[1]], bitsize, low, high).tolist()
    #         for ppair in parents:
    #             child = cross_parents(rpop[ppair[0]], rpop[ppair[1]], bitsize)
    #
    #             newgen.append(cross_parents(rpop[ppair[0]], rpop[ppair[1]], bitsize))
    #
    #         # Select top10
    #         t10 = roulette_select(rpop, tfunc, 64)[:5]
    #         genlist.append([])
    #         for ppair in t10:
    #             genlist[j].append(rpop[ppair[0]])
    #             genlist[j].append(rpop[ppair[1]])
    #
    #         genlist[j] = np.array(genlist[j])
    #
    #         # genlist.append(rpop)
    #         rpop = np.array(newgen)
    #         parents = roulette_select(np.array(newgen), tfunc, bitsize)
    #
    #     # genlist.append(rpop)
    #     genarr = np.empty((size[0], epochs), dtype=object)
    #
    #     k = 0
    #     for i in genlist:
    #         for j in range(i.shape[0] - 1):
    #
    #             strbit = "".join(str(s) for s in i[j])
    #             # print("~~~~~~")
    #             # print("strbit: %s" % strbit)
    #             # print("i: %s, k: %s" % (j, k))
    #             # print(genarr[j, k])
    #             genarr[j, k] = strbit
    #             # print(genarr[j, k])
    #         k += 1
    #
    #     genarr = genarr.T
    #     # print(genarr)
    #     # print(Ndbit2float(rpop[0], bitsize))
    #
    #     dataind = 5
    #     with open("GAmult_%s_tfunc%s_bsize%s_sim%s.txt" % (dataind, tfunc.__name__, bitsize, sim), 'w') as f:
    #         f.write(';'.join([str(i) for i in range(genarr.shape[0])]) + "\n")
    #         genarr = genarr.T
    #         for i in range(genarr.shape[0]):
    #             f.write(";".join([str(item) for item in genarr[i]]) + "\n")

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



