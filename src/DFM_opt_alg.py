from typing import Callable, Union, Iterable, Optional

from population_initatilisation import *
from selection_funcs import *
from cross_funcs import *
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


def mutate(bit, bitsize, **kwargs):
    global bdict

    bitc = bit.copy()
    nbits = int(bitc.size/bitsize)


    mutate_coeff = int(bit.size/bitsize)
    if "mutate_coeff" in kwargs:
        mutate_coeff = kwargs["mutate_coeff"]
    # mutations = np.random.randint(nbits * (1 + bdict[bitsize][1]), bit.size, mutate_coeff)
    mutations = np.random.choice(np.arange(nbits * (1 + bdict[bitsize][1]), bit.size), mutate_coeff, replace=False)
    # Speed up?
    for mutation in mutations:
        if bitc[mutation]:
            bitc[mutation] = 0
        else:
            bitc[mutation] = 1

    return bitc


def full_mutate(bit, bitsize, **kwargs):
    global bdict

    bitc = bit.copy()

    mutate_coeff = int(bit.size/bitsize)
    if "mutate_coeff" in kwargs:
        mutate_coeff = kwargs["mutate_coeff"]
    # mutations = np.random.randint(nbits * (1 + bdict[bitsize][1]), bit.size, mutate_coeff)
    mutations = np.random.choice(np.arange(0, bit.size), mutate_coeff, replace=False)
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
        self.cross: Callable = equal_prob_cross
        self.mutation: Callable = mutate

        # self.seed: Callable = self.none

        self.epochs = None
        self.shape = (0, 0)
        self.elitism = 10
        self.save_top = 10

        self.results: list = []

        self.dolog = 0
        self.log = {}

        self.min = np.NAN

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def __repr__(self):
        return ""

    def __getitem__(self, item):
        return self.results[item]

    def run(self, selargs: dict = {},
            cargs: dict = {}, muargs: dict = {},
            epochs: int = 100, verbosity: int = 1):
        """
        :param cargs:
        :param muargs:
        :param seedargs:
        :param selargs:
        :param epochs:
        :param verbosity:
        :return:
        """

        if len(self.pop) == 0:
            self.init_pop()

        self.epochs = epochs

        selargs["fx"] = self.tfunc
        selargs["bitsize"] = self.bitsize
        parents = self.select(self.pop, **selargs)

        # if self.seed.__name__ == "none":
        #     self.epochs = int(np.floor(np.log2(self.shape[0])))


        for epoch in range(self.epochs):
            # seedargs["low"] = self.min
            newgen = []
            if verbosity:
                print("%s/%s" % (epoch + 1, self.epochs))

            cargs["bitsize"] = self.bitsize
            muargs["bitsize"] = self.bitsize

            for ppair in parents[:self.elitism]:
                # child1, child2 = self.cross(self.pop[ppair[0]], self.pop[ppair[1]], **cargs)
                child1, child2 = self.cross(self.pop[ppair[0]], self.pop[ppair[1]], **cargs)
                newgen.append(child1)
                newgen.append(child2)

            for ppair in parents[self.elitism:]:
                child1, child2 = self.cross(self.pop[ppair[0]], self.pop[ppair[1]], **cargs)
                # print("~~~~~~")
                # print(Ndbit2float(child, 32) - Ndbit2float(self.mutation(child, **muargs), 32))
                # print(Ndbit2float(child1, self.bitsize), Ndbit2float(child2, self.bitsize))
                newgen.append(self.mutation(child1, **muargs))
                newgen.append(self.mutation(child2, **muargs))

            # Select top10
            t10 = self.select(self.pop, **selargs)[:self.save_top]
            self.genlist.append([])
            for ppair in t10:
                self.genlist[epoch].append(self.pop[ppair[0]])
                self.genlist[epoch].append(self.pop[ppair[1]])

            self.genlist[epoch] = np.array(self.genlist[epoch])

            # genlist.append(rpop)
            self.pop = np.array(newgen)
            parents = self.select(np.array(newgen), **selargs)

            if self.dolog:
                if self.dolog == 2:
                    pass
                elif self.dolog == 1:
                    pass

            # y = np.apply_along_axis(self.tfunc, 1, Ndbit2float(self.pop, self.bitsize))
            # self.min = np.min(y)

        self.results = self.genlist

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
            ['uniform', 'cauchy', 'nbit']
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
        elif method == "nbit":
            self.pop = bit8(**kwargs)
        else:
            self.pop = method(**kwargs)

        self.shape = (self.pop.shape[0], self.pop.shape[1]/self.bitsize)

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
        self.shape = (self.pop.shape[0], self.pop.shape[1] / self.bitsize)
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
        return self.results

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

        data = list(Fileread(path=path, dtype="str", delimiter=";")().values())

        for i in range(len(data)):
            data[i] = np.delete(data[i], np.where(data[i] == "None"))
            resmat = np.empty((len(data[i]), len(data[i][0])), dtype=np.uint8)
            for x in range(len(data[i])):
                for b in range(len(data[i][x])):
                    resmat[x, b] = int(data[i][x][b])

            data[i] = resmat

        self.results = data

        return self.results

    def get_numeric(self, bit2num: Callable = Ndbit2float, **kwargs):
        """
        Convert results list with np.ndarrays of dimension mx1 to numeric data
        using provided method or builtin routine for multi variable
        float conversion.
        Provided method needs to accept np.ndarray with binary information
        stored as dtype np.uint8 and return np.ndarray of shape nx1 contaning
        numeric data of float, int.
        Example
        def b2int(bit: np.ndarray) -> np.ndarray:
            '''
            Conversion of m x n (big endian) bit array to integers.
            :param bit2num: m x n ndarray of numpy integers (0, 1) representing a bit
            :return: m x n ndarray of integers
            '''
            m, n = bit.shape
            a = 2 ** np.arange(n)
            return bit @ a
        :param bit2num:
            Binary to numeric conversion routine
        :param kwargs:
            kwargs for bit2num
        :return: numeric values for results of the loaded GA data
        """
        return [bit2num(i, **kwargs) for i in self.results]

    def logdata(self, verbosity: int = 2):
        """
        Initiate self.log to verbosity to save extra information per epoch in a
        dictionary with key epoch and value a dict with for verbosity level 2
        {
         time: float, #time of one epoch
         ranking: list of list, # result from self.select
         fitness: list of float, # fitness values of all input values ordered as ranking
         value: list of np.ndarray, # genlist with all bitvalues.
         value_top: list of np.ndarray, # genlist with bitvalues selected by top.
         value_num: list of np.ndarray, # genlist with all values converted to numeric values.
        }

        and for verbosity level 1
        {
         time: float, #time of one epoch
         ranking: list of list, # result from self.select
         value: list of np.ndarray, # genlist with all values.
        }

        The resulting dict will look like the following
            {
             0: epoch_log_dict,
             1: epoch_log_dict,
             ...
             n: epoch_log_dict
            }

        Save log to txt with the log2txt method
        Get log by calling the get_log method or by assigning a variable
        to geneticalgorithm.log

        :return: None
        """
        self.log = {}
        self.dolog = verbosity

        return None

    def get_log(self):
        return self.log

    def log2txt(self, path):
        return None

    @staticmethod
    def none(*args, **kwargs):
        return None


if __name__ == "__main__":
    tsart = time()


    size = [10, 2]
    low, high = -1, 1
    bitsize = 9
    tfunc = wheelers_ridge

    # epochs = int(np.floor(np.log2(size[0])))
    epochs = 10

    ga = genetic_algoritm(bitsize=bitsize)
    ga.init_pop("nbit", shape=size, bitsize=bitsize)

    # ga.seed = uniform_bit_pop_float
    ga.set_cross(full_double_point)
    ga.set_mutate(full_mutate)

    ga.save_top = size[0]

    ga.target_func(tfunc)
    ga.run(epochs=epochs, muargs={"mutate_coeff": 2}, selargs={"nbit2num": ndbit2int})
    ga.save_results("wheeler8bit1.txt")
    # print(Ndbit2float(ga[-1], 64))
    # print(ga.load_results("GAmult_5_tfuncwheelers_ridge_bsize64_sim0.txt"))
    # print(ga.get_numeric(bitsize=64)[-1])

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
