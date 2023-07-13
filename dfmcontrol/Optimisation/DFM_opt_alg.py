from typing import Callable, Union, Iterable, Optional

import numpy as np
import pickle

from time import time

try:
    from dfmcontrol.pop import *
    # from .gradient_descent import gd
    from dfmcontrol.selection_funcs import *
    from dfmcontrol.cross_funcs import *
    from dfmcontrol.Mathematical_functions import *
    from dfmcontrol.mutation import *
    from dfmcontrol.log import log

except ImportError:
    from dfmcontrol.Utility.pop import *
    # from dfmcontrol.gradient_descent import gd
    from dfmcontrol.Utility.selection import *
    from dfmcontrol.Utility.crossover import *
    from dfmcontrol.Utility.mutation import *
    from dfmcontrol.Mathematical_functions import *
    from dfmcontrol.Log.log import log

bdict = {8: [1, 4, 3], 16: [1, 5, 10], 32: [1, 8, 23], 64: [1, 11, 52],
         128: [1, 15, 112], 256: [1, 19, 236]}


# def geneticalg(fx: Callable, pop: np.ndarray, max_iter: int, select: Callable,
#                cross: Callable, IEEE_mutate: Callable):
#     """
#     :param fx:
#     :param pop:
#     :param max_iter:
#     :param select:
#     :param cross:
#     :param IEEE_mutate:
#     :return:
#     """
#     # fx = np.vectorize(fx)
#     for _ in range(max_iter):
#         # Parents function should return pairs of parent indexes in pop
#         parents = select(b2int(pop), fx)
#         # Apply the cross function to all parent couples
#         children = [cross(pop[p[0]], pop[p[1]]) for p in parents]
#         # Mutate the population (without elitism)
#         pop = IEEE_mutate(children)
#
#     return pop


class genetic_algoritm:
    """
    Genetic Algorithm functions

    :param dtype: data type of the population
    :param bitsize: size of the bit array
    :param endianness: endianness of the bit array

    The genetic algorithm is a stochastic search algorithm that is used to find the global minimum (or maximum) of a function.
    In this case the algortihm is initialised with a population of random bit arrays with size bitsize. The population is then
    iterated over and the fitness of each bit array is calculated, the fitness is then used to select the parents for the next
    generation by the selection algorithm defined by genetic_algortihm.select, the algortihm used can be changed by calling the
    set_select() method or setting it manually by assigning genetic_algorthim.select to a different function.
    The parents are then crossed over and mutated to create the next generation with the crossing and mutation algorithm defined by
    genetic_algorithm.cross and .IEEE_mutate respectively. The process is repeated until the maximum number of iterations is reached.
    The results are stored in the results attribute and can be accessed by calling genetic_algorithm.get_results() or by indexing
    the genetic_algorithm object. A log of all fitness, rank, output and time values is also stored in the log attribute and can be accessed
    by calling genetic_algorithm.get_log().<attr>.<value> or by indexing the genetic_algorithm.log.<attr>.<value> object. This log
    can be saved to a .pickle file by calling genetic_algorithm.save_log() or by calling genetic_algorithm.save_log() with a file name.
    """

    def __init__(self, dtype: str = "float", bitsize: int = 32,
                 endianness: str = "big"):
        self.dtype = dtype
        self.bitsize: int = bitsize
        self.endianness: str = endianness

        self.genlist: list = []
        self.pop: np.ndarray = np.array([])
        self.initial_pop: np.ndarray = np.array([])

        self.tfunc: Callable = self.none
        self.targs: dict = {}

        if not is_decorated(self.tfunc):
            self.tfunc = tfx_decorator(self.tfunc)

        self.select: Callable = roulette_selection
        self.cross: Callable = equal_prob
        self.mutation: Callable = mutate

        self.mode: str = "optimise" # or mnimise
        self.modebool: bool = True if self.mode == "optimise" else False

        self.tstart = time()

        self.epochs = None
        self.shape: tuple = (0, 0)
        self.elitism: int = 5
        self.save_top: int = 10

        self.targetfx: Union[Iterable, float] = self.tfunc.minima["x"]
        self.results: list = []

        self.dolog: int = 2
        self.b2n: Callable = ndbit2int
        self.b2nkwargs: dict = {}
        self.log: "log" = log(self.pop, self.select, self.cross, self.mutation,
                              self.b2n, self.elitism, self.save_top,
                              self.bitsize, self.b2nkwargs)

    def __call__(self, *args, **kwargs):
        """
        Call the genetic algorithm with the given arguments

        :param args: arguments for the target function

        :param kwargs: keyword arguments for the target function

        :return: None
        """
        self.run(*args, **kwargs)

    def __repr__(self):
        return ""

    def __getitem__(self, item: int) -> Union[dict, list]:
        """
        Get the results of the genetic algorithm

        :param item: index of the result

        :return: result
        """
        return self.results[item]

    def run(self, selargs: dict = {},
            cargs: dict = {}, muargs: dict = {},
            epochs: int = 100, verbosity: int = 1,
            runcond: bool = None):
        """
        Run the genetic algorithm

        :param cargs: arguments for the cross function

        :param muargs: arguments for the mutation function

        :param selargs: arguments for the selection function

        :param epochs: number of iterations

        :param verbosity: verbosity level

        :return: None
        """
        # print(self.pop, self.initial_pop)
        if len(self.pop) == 0:
            self.init_pop()

        if not is_decorated(self.tfunc):
            self.tfunc = tfx_decorator(self.tfunc)


        self.epochs = epochs
        self.epoch = 0

        if runcond is None:
            runcond = "self.epoch < self.epochs"

        self.targetfx: Union[Iterable, float] = self.tfunc.minima["x"]

        selargs["fx"] = self.tfunc
        selargs["bitsize"] = self.bitsize
        selargs["b2n"] = self.b2n
        selargs["b2nkwargs"] = self.b2nkwargs
        selargs["verbosity"] = verbosity
        selargs["mode"] = self.modebool

        self.tfunc.set_dimension(self.shape[1])

        parents, fitness, p, fx = self.select(self.pop, **selargs)

        # if self.seed.__name__ == "none":
        #     self.epochs = int(np.floor(np.log2(self.shape[0])))

        if self.dolog:
            # Highest log level
            rank = []
            for ppair in parents:
                rank.append(self.pop[ppair[0]])
                rank.append(self.pop[ppair[1]])
            rank = np.asarray(rank)

            if self.dolog == 2:

                self.log.ranking.update(rank, fx, self.tfunc.minima["x"], self.tfunc.minima["fx"])
                self.log.time.update(time() - self.tstart, 0)
                self.log.selection.update(parents, p, fitness)

                if len(self.log.add_logs) > 0:
                    for l in self.log.add_logs:
                        l.update(data=self.pop)

                    self.log.sync_logs()

            elif self.dolog == 1:
                self.log.ranking.update(rank, self.tfunc.minima["x"])
                self.log.time.update(time() - self.tstart)


        while self.condinterpreter(runcond):
            newgen = []

            # print(self.tfunc.minima["fx"])
            # print(self.tfunc(self.b2n(self.pop, **self.b2nkwargs)))
            # print(np.abs(self.log.ranking.distancefx[-1]))
            # print("------------------")


            if verbosity:
                print("%s/%s" % (self.epoch + 1, self.epochs))
                print("Distance to sol: %s" % np.min(np.abs(self.log.ranking.distancefx[-1])))
                print("Group distance: %s" % np.average(np.abs(self.log.ranking.distancefx[-1])))
                print("Best fitness: %s" % np.max(self.log.selection.fitness[-1]))

            cargs["bitsize"] = self.bitsize
            muargs["bitsize"] = self.bitsize

            for ppair in parents[self.elitism:]:
                child1, child2 = self.pop[ppair[0]], self.pop[ppair[1]]
                # child1, child2 = self.cross(self.pop[ppair[0]],
                #                             self.pop[ppair[1]], **cargs)

                newgen.append(child1)
                newgen.append(child2)

            for ppair in parents[:self.elitism]:
                child1, child2 = self.cross(self.pop[ppair[0]],
                                            self.pop[ppair[1]], **cargs)
                newgen.append(self.mutation(child1, **muargs))
                newgen.append(self.mutation(child2, **muargs))

            # Select top10
            t10 = parents[:self.save_top]
            self.genlist.append([])
            for ppair in t10:
                self.genlist[self.epoch].append(self.pop[ppair[0]])
                self.genlist[self.epoch].append(self.pop[ppair[1]])

            self.genlist[self.epoch] = np.array(self.genlist[self.epoch])

            # genlist.append(rpop)
            self.pop = np.array(newgen)
            parents, fitness, p, fx = self.select(np.array(newgen), **selargs)

            if self.dolog:
                # Highest log level
                rank = np.zeros(self.pop.shape)
                if self.dolog == 2:
                    rankind = np.argsort(fitness)

                    j = 0
                    for i in rankind:
                        rank[j] = self.pop[i]
                        j += 1

                    # print(fx, self.tfunc.minima["fx"])
                    
                    self.log.ranking.update(rank, fx, self.tfunc.minima["x"],
                                            self.tfunc.minima["fx"])
                    self.log.time.update(time() - self.tstart, self.tfunc.calculations)
                    self.log.selection.update(parents, p, fitness)
                    self.log.value.update(self.pop, self.genlist[self.epoch])

                    # if additional logs added by appending them after initiation of self.log
                    # go through them and update with the population
                    # other data can be found within other logs and data
                    # can be added by using global statements or other.
                    if len(self.log.add_logs) > 0:
                        for l in self.log.add_logs:
                            l.update(data=self.pop)

                        self.log.sync_logs()

                elif self.dolog == 1:
                    self.log.ranking.update(rank, self.tfunc.minima["x"])
                    self.log.time.update(time() - tsart)

                    self.log.logdict[self.epoch] = {"time": time() - self.tstart,
                                               "ranking": rank,
                                               "value": self.pop}

            self.epoch += 1

        self.results = self.genlist

    def run_threaded(self, threads, **kwargs):
        """
        Run the genetic algorithm in multiple threads

        :param threads: number of threads

        :param kwargs: arguments for the run function

        :return: None
        """
        self.threads = threads
        # self.threadpool = ThreadPool(threads)
        # self.threadpool.map(self.run, [kwargs] * threads)


    def init_pop(self, method: Union[str, Callable] = "uniform", **kwargs):
        """
        set self.pop to an array generated by predefined routines or usermade method.

        :param method: Optional[str, callable]
            if str method will be initialised by routines included in population_initialisation.py
            the str should match the init method, so uniform -> 'uniform' and
            cauchy 'cauchy' etc
            full list of usable args:
            ['uniform', 'cauchy', 'normal']
            if callable the population will be the return value of given function.
            A population function should return a mx1 numpy array of bits, to convert
            floating point values to approved bit values use the float2Ndbit function
            included in helper.py

        :param kwargs: kwargs for init method, generally the first argument is the shape
            of the population, the second is the bitsize for the population (auto filled)
            followed by routine specific kwargs (see the wiki or pop.py).
            
        :param kwargs[shape]: Tuple of ints (m, n) where m is the number of individuals
            and n is the number of genes per individual.
        :param kwargs[bitsize]: optional int, number of bits per individual.
        :param kwargs[boundaries]: tuple of floats, boundaries for the population
            list of ints or floats [lower, upper]. Only used for uniform and normal
            distributions.
    
        :return: None
        """
        if "bitsize" not in kwargs:
            kwargs["bitsize"] = self.bitsize
        
        if self.b2n is ndbit2int and method in ["uniform", "cauchy", "normal"]:
            if method == "uniform":
                self.pop = uniform_bit_pop(**kwargs)
            elif method == "cauchy":
                self.pop = cauchy_bit_pop(**kwargs)
            elif method == "normal":
                self.pop = bitpop(**kwargs)
        elif self.b2n is Ndbit2floatIEEE754 and method in ["uniform", "cauchy", "normal"]:
            if method == "uniform":
                self.pop = uniform_bit_pop_IEEE(**kwargs)
            elif method == "cauchy":
                self.pop = cauchyrand_bit_pop_IEEE(**kwargs)
            elif method == "normal":
                self.pop = normalrand_bit_pop_IEEE(**kwargs)
        else:
            self.pop = method(**kwargs)

        self.log.pop = self.pop
        self.shape = (self.pop.shape[0], self.pop.shape[1] / self.bitsize)

        self.initial_pop = self.pop

        return None

    @property
    def set_pop(self, reset: bool = True):
        """
        Set population (self.pop) to provided ndarray of bits.

        :param pop: np.ndarray of shape mx1, with bits in numpy arrays of
         dtype: np.uint8 like: [[0, 1, ... ,0, 1], [0, 1, ... ,0, 1], [0, 1, ... ,0, 1]]

        :param reset: bool, if True the initial population will be set to the provided pop.

        :return: None
        """
        self.pop = pop
        self.log.pop = self.pop
        self.shape = (self.pop.shape[0], self.pop.shape[1] / self.bitsize)

        if reset:
            self.initial_pop = self.pop

        return None

    @property
    def get_pop(self):
        """
        Return a copy of population (self.pop)

        :return: self.pop.copy()
        """
        return self.pop.copy()

    def target_func(self, target, targs: dict = None):
        """
        Set target function to be used in the optimisation.

        :param target: Callable function to be used as target function,
         should take an array of [x1, x2, ... , xn] (array of float)
         and return f([x1, x2, ... ,xn]) (float).
        :param targs: dict of kwargs for target function.

        :return: None
        """

        self.tfunc = target
        self.targs = targs
        return None

    def get_results(self):
        """
        Return a copy of results (self.results)

        :return: self.results.copy()
        """
        return self.results.copy()

    def set_cross(self, cross: Callable):
        """
        Set the cross method used in the GA, method should take 2 arguments:
        parent1 and parent2 (both np.ndarray of dim 1 with dtype np.uint8) + optional kwargs
        and return a single numpy array with binary value of the resulting child
        from p1 and p2.

        :param cross:
            Method to cross binary data p1 and p2 to form child.

        Example method:
         def cross(p1, p2, bitsize, **kwargs):
            ''' Cross two parents to form a child ''' \n
            # Take the first half of p1 and cross it with the other half of p2 \n
            return np.concatenate([p1[:int(1/2 * bitsize)], p2[int(1/2 * bitsize):]])

        It is recommended to add **kwargs to the provided method to be able to
        handle excess arguments.

        :return: None
        """
        self.cross = cross
        return None

    def set_select(self, select: Callable):
        """
        Set the parent selection method used in GA, method should take an
        np.ndarray of dim mx1 as an argument + kwargs. The method should return
        4 arrays:
         parent indexes, fitness, probability of selection and function results
        Of which only the first one is required for the algorithm, the other 3
        arrays are passed on to the log.select attribute and can be empty
        arrays.

        :param select:
            Method to select parent combination by returning their indexes in the
            population self.pop.

        Example method: \n
          def select(pop, **kwargs):
            '''Return completely random combinations in pop array''' \n
            return np.random.choice(range(pop.shape[0]), 2, replace=False)

         It is recommended to add **kwargs to the provided method to be able to
         handle excess arguments.

        :param optional fitness:
          Method to determine the fitness of a population which will be logged
          in self.log.

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
            Method to IEEE_mutate a single np.ndarray of bits

        Example method:
         def IEEE_mutate(bit, **kwargs):
            ''' Mutate a single bit ''' \n
            # select a random point in the bit \n
            ind = np.random.randint(0, bit.size)
            # if 1 turn 0 else turn 1. \n
            if bit[ind]:
                bit[ind] = 0
            else:
                 bit[ind] = 1
            # return the mutated bit
            return bit

        It is recommended to add **kwargs to the provided method to be able to
        handle excess arguments.

        :return None:
        """

        self.mutation = mutation
        return None

    def save_results(self, path: str):
        """
        Save the results of the GA to a .txt file.
        :param path: str
        :return: None
        """
        genarr = np.empty((self.genlist[0].shape[0] - 1, self.epochs),
                          dtype=object)
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
        """
        Load the results of a GA from a .txt file.

        :param path: str

        :return: None
        """
        from dfmcontrol.AdrianPackv402.Fileread import Fileread

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

    def get_numeric(self, bit2num: Callable = None, target: list = None,
                    **kwargs):
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
        if bit2num == None:
            bit2num = self.b2n

        if target == None:
            target = self.results

        return [bit2num(i, **kwargs) for i in target]

    def logdata(self, verbosity: int = 2):
        """
        Initiate log class to verbosity to save extra information per epoch in a
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

        The log class saves all the data to seperate methods in addition to this
        logdict, see the log class documentation.

        *Logdict is deprecated and will be removed in future versions*

        :return: None
        """
        self.log = log(self.pop, self.select, self.cross, self.mutation,
                       self.b2n, self.elitism, self.save_top,
                       self.bitsize, self.b2nkwargs)

        self.dolog = verbosity

        return None

    def save_log(self, path: str = ""):
        """
        Save the logdict to a .pickle file.

        :param path: str

        :return: None
        """
        if path == "":
            date_str = self.log.creation.__str__().replace(":", "_")
            path = "GAlog%s" % date_str + ".pickle"
        with open(path, "wb") as f:
            pickle.dump(self.log, f)

        print("Log saved to: %s" % path)

    def load_log(self, path: str, copy: bool = True):
        """
        Load a logdict from a .pickle file.

        :param path: path to .pickle file

        :param copy: If True, the logdict will be copied to the current GA instance.

        :return: None
        """

        with open(path, "rb") as f:
            old_log = pickle.load(f)

        if copy:
            self.log = log(old_log.pop, old_log.select, old_log.cross,
                           old_log.mutation,
                           old_log.b2n, old_log.elitism, old_log.savetop,
                           old_log.bitsize, old_log.b2nkwargs)

            self.log.creation = old_log.creation

            self.log.time.data = old_log.time.data
            self.log.time.epoch = old_log.time.epoch
            self.log.time.calculation = old_log.time.calculation

            self.log.ranking.data = old_log.ranking.data
            self.log.ranking.epoch = old_log.ranking.epoch
            self.log.ranking.ranknum = old_log.ranking.ranknum
            self.log.ranking.effectivity = old_log.ranking.effectivity
            self.log.ranking.distancex = old_log.ranking.distancex
            self.log.ranking.distancefx = old_log.ranking.distancefx
            self.log.ranking.bestsol = old_log.ranking.bestsol
            self.log.ranking.result = old_log.ranking.result
            self.log.ranking.bestresult = old_log.ranking.bestresult

            self.log.selection = old_log.selection.copy()

            self.log.value.data = old_log.value.data
            self.log.value.epoch = old_log.value.epoch
            self.log.value.value = old_log.value.value
            self.log.value.numvalue = old_log.value.numvalue
            self.log.value.topx = old_log.value.topx

            if len(old_log.add_logs) > 0:
                for l in old_log.add_logs:
                    self.log.add_logs.append(l)
                self.log.sync_logs()

        else:
            self.log = old_log

        return self.log

    def condinterpreter(self, cond: str):
        """
        Interpret a condition string and return a list of conditions.


        :param cond:
        :return:
        """

        cndsplit = cond.split(" ")
        conditions = []

        for i in range(len(cndsplit)):
            if cndsplit[i] in ["<", ">", "==", "!=", "<=", ">="]:
                conditions.append([cndsplit[i - 1], cndsplit[i], cndsplit[i + 1]])

        # For all conditions construct  lambda function to test the condition
        for i in range(len(conditions)):
            conditions[i] = eval(conditions[i][0] + conditions[i][1] + conditions[i][2])

        for i in conditions:
            if not i:
                return False

        return True

    def reset(self, reset_pop = True):
        """
        Reset the GA instance to the initial state.

        :param reset_pop: If False, the population will be reset to the initial population.
        :return: None
        """
        self.genlist: list = []
        self.pop = self.initial_pop

        self.logdata(2)

        if reset_pop:
            self.pop: np.ndarray = np.array([])

        self.log.pop = self.pop


    @staticmethod
    @tfx_decorator
    def none(*args, **kwargs) -> None:
        """
        Dummy function for no transformation.

        :param args: args

        :param kwargs: kwargs

        :return: None
        """
        return None


if __name__ == "__main__":
    tsart = time()


    def inv_ackley(x):
        return booths_function(x)


    d = 2

    size = [25, d]
    low, high = -10, 10
    bitsize = 16
    tfunc = ackley
    # epochs = int(np.floor(np.log2(size[0])))
    epochs = 10

    iteration = 10

    p = 0.01

    k = np.e
    ga = genetic_algoritm(bitsize=bitsize)

    print(ga.log.creation)

    # ga.optimumfx = np.full(d, -2.903534)

    ga.b2n = ndbit2int
    ga.logdata(2)

    ga.init_pop("uniform", shape=[size[0], size[1]], bitsize=bitsize,
                boundaries=[low, high], factor=10)
    ga.b2nkwargs = {"factor": 10}

    ga.elitism = 2

    # ga.seed = uniform_bit_pop_IEEE
    ga.set_cross(equal_prob)
    ga.set_mutate(mutate)
    ga.set_select(rank_selection)

    ga.save_top = 10

    ga.target_func(tfunc)

    ga.run(epochs=epochs, muargs={"mutate_coeff": 3},
           selargs={"nbit2num": ndbit2int,
                    "k": k, "fitness_func": no_fitness,
                    "allow_duplicates": True},
           verbosity=0, runcond="min(np.abs(self.log.ranking.distancefx[-1])) > 0.1")

    ga.save_log("Booth16b_p%s.pickle" % iteration)
    iteration += 0

    lg = ga.log

    print("Distancefx: ", min(np.abs(lg.ranking.distancefx[-1])))
    print("Best solution: %s" % lg.ranking.bestsol)
    print("Computations %s" % lg.time.calculation)
    print("t: ", time() - tsart)

    lg.value.animate2d(ackley, low, high, fitness=lg.selection.fitness)

