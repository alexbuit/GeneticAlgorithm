import random

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys

from typing import Union, Callable, List
from time import sleep, time

from DFM_opt_alg import genetic_algoritm, full_mutate
from src.cross_funcs import full_equal_prob
from src.helper import ndbit2int
from src.selection_funcs import *
from src.log import log_object

from AdrianPackv402.Helper import compress_ind

## global variables
individuals: int = 25
points_per_indv: int = 100
points_stability_test = 500

runtime = 4 # runtime in seconds
epoch = 0  # ??

# Hardcoded value for 39ch mirror
individual_size = 39

test_setup = True

# Logs intensity of stability test and algorithm
# array of dim 4, saved as dim 5 after run [epochs, 2: [stability, test], individuals, 3: [intensity, time, bin combination], sample size]
# sample size for algorithm -> points_per_indiv
# sample size of stability test -> points_stability_test (uses floating point average to compress to length sample size algorithm)
intens_blueprint = np.zeros(shape=[2, individuals, 3, points_per_indv])
intens: List[np.ndarray] = [intens_blueprint.copy()]
# Lookup table for voltages indexed in array "intens"
individual_table_blueprint = np.zeros(shape=[individuals, 2, individual_size])
individual_table: List[np.ndarray] = [np.zeros(shape=[individuals, 2, individual_size])]

"""How to find a voltage:
# Find the number of the individual in intens;
idx = intens[epoch, testtype, individual, 3, sample] ## sample # doesnt matter since for each sample the combination is the same
# Use this index to find the voltage combination in individuals;
indv = individuals[epoch][idx, testtype, :] ## this is the voltage combination
# Convert to binary using the ga.b2n() method;
ga.b2n(indv, **ga.b2nkwargs)
"""

# time intensity logs the intensity each time read_pm is called and logs the according time
# array of dim 1 saved as dim 2 after run [entries, 2: [intens, time]]
time_intens: List[np.ndarray] = [np.zeros(2)]

optimum: float = 100

t0 = time()

if __name__ == "__main__" and not test_setup:
    print("start")

    # Opening the mirror handle
    import okotech_lib.okodm_sdk.python.okodm_class as oko
    import sys

    handle = oko.open("MMDM 39ch,15mm",
                      "USB DAC 40ch, 12bit")  # handle for other functions

    if handle == 0:
        sys.exit(("Error opening OKODM device: ", oko.lasterror()))

    # Get the number of channels
    n = oko.chan_n(handle) # Should be 39

    print("n channels= ", n)

    ## Opening the DMMM
    import pyvisa
    from ThorlabsPM100 import ThorlabsPM100

    rm = pyvisa.ResourceManager()

    print(rm.list_resources())
    inst = rm.open_resource('USB0::0x1313::0x8078::P0000874::INSTR', timeout=1)

    power_meter = ThorlabsPM100(inst=inst)

    def read_pm():
        global time_intens
        time_intens.append(np.ndarray([power_meter.read, time() - t0]))
        return time_intens[-1][1]

    def set_voltage(voltages):
        try:
            if not oko.set(handle, voltages):
                sys.exit("Error writing to OKODM device: " + oko.lasterror())
        except:
            voltages = np.zeros(shape=n)
            if not oko.set(handle, voltages):
                sys.exit("Error writing to OKODM device: " + oko.lasterror())

elif test_setup:
    n = 39

    def read_pm():
        global time_intens
        time_intens.append(np.array([-np.exp(random.random()/ 100), time() - t0]))
        return time_intens[-1][1]

    def set_voltage(voltages):
        pass




## Additional functions for the algorithm

def tfmirror(*args, **kwargs):
    """
    Takes an individual and sets the mirror to the corresponding voltage

    :param individual: Individual to be set
    :param handle: Handle for the mirror
    :param n: Number of channels
    :return: None
    """
    global intens, optimum, epoch, points_per_indv

    if "pop" in kwargs:
        pop = kwargs["pop"]
    else:
        pop: np.ndarray = args[0]

    b2n: Callable = kwargs["b2n"]
    b2nkwargs: dict = kwargs["b2nkwargs"]

    ppi: int = kwargs.get("points_per_indv", 100)
    stability: bool = kwargs.get("stability", False)

    num_pop = b2n(pop, **b2nkwargs)

    # Values are normalised, how much time does this take?
    for indiv in num_pop:
        for var in indiv:
            try:
                assert -1 < var < 1
            except AssertionError:
                voltages = np.zeros(shape=n)
                set_voltage(voltages)

                raise ValueError("Input voltage can not be > 1 or < -1")

    avg_read = np.zeros(num_pop.shape[0])

    i = 0
    for indiv in num_pop:
        voltages = indiv

        set_voltage(voltages)

        # Read the power meter
        compress_this = np.zeros([2, ppi])
        for j in range(ppi):
            if stability:
                compress_this[0, j] = read_pm()
                compress_this[1, j] = time() - t0
            else:
                intens[epoch][1, i, 0, j] = read_pm()
                intens[epoch][1, i, 1, j] = time() - t0
                intens[epoch][1, i, 2, j] = i
                individual_table[epoch][i, 1, :] = indiv

        if stability:
            this_compressed_intens = compress_ind(compress_this[0, :], points_per_indv)[0]
            this_compressed_time = compress_ind(compress_this[1, :], points_per_indv)[0]
            intens[epoch][0, i, 0, :] = this_compressed_intens
            intens[epoch][0, i, 1, :] = this_compressed_time
            intens[epoch][0, i, 2, :] = np.full(points_per_indv, i, dtype=int)
            individual_table[epoch][i, 0,:] = indiv

        if stability:
            avg_read[0] = np.average(intens[epoch][0, i, 0, :])
        else:
            avg_read[i] = np.average(intens[epoch][1, i, 0, :])/optimum

        i += 1
        # Test out on DFM and append intensity
        # Divide intensity by global var optimum

    return avg_read


def select(*args, **kwargs):
    global intens, optimum, points_per_indv, epoch, points_stability_test

    pop = args[0]

    kwargs["points_per_indv"] = points_per_indv
    kwargs["stability"] = False

    fitness = tfmirror(*args, **kwargs)

    # probability paramter for rank selection
    prob_param = 1.9
    if "p" in kwargs:
        prob_param = kwargs["p"]

    fit_rng = np.flip(np.argsort(fitness))

    # Test the best individual for stability
    kwargs["points_per_indv"] = points_stability_test
    kwargs["stability"] = True
    kwargs["pop"] = pop[fit_rng[0]]

    tfmirror(**kwargs)

    p = (prob_param * (1 - prob_param)**(np.arange(1, fitness.size + 1, dtype=float) - 1))
    p = p/np.sum(p)

    selection_array = np.zeros(fit_rng.shape)
    for i in range(fitness.size):
        selection_array[fit_rng[i]] = p[i]

    pind = []
    rng = np.arange(0, fitness.size)

    for i in range(int(fitness.size / 2)):
        if selection_array.size > 1:
            try:
                par = np.random.choice(rng, 2, p=selection_array,
                                       replace=False)
            except ValueError:
                if kwargs["verbosity"] == 1:
                    print("Value error in selection, equal probability")

                selection_array = np.full(selection_array.size,
                                          1 / selection_array.size)
                par = np.random.choice(rng, 2, p=selection_array,
                                       replace=False)

            pind.append(list(sorted(par).__reversed__()))

    set_voltage(np.zeros(shape=n))

    return pind, fitness, p, fitness


class log_intensity(log_object):
    def __init__(self, b2num, bitsize, b2nkwargs, *args, **kwargs):
        super().__init__(b2num, bitsize, b2nkwargs, *args, **kwargs)
        self.intensity = []
        self.indivuals = []

    def update(self, data, *args):
        global intens
        self.data.append(data)
        self.epoch.append(len(self.data))
        self.intensity.append(*args[0])


    def __copy__(self):
        log_intens_c = log_intensity(self.b2n, self.bitsize, self.b2nkwargs)
        log_intens_c.intensity = self.intensity
        log_intens_c.data = self.data
        log_intens_c.indivuals = self.indivuals
        log_intens_c.epoch = self.epoch

        return log_intens_c

    def plot(self, epoch: Union[slice, int] = slice(0, None),
             individual: Union[slice, int] = slice(0, None),
             data: Union[slice, int] = slice(0, None),
             fmt_data: str = "average", linefmt="scatter"):

        if fmt_data.lower() == "raw":
            int_mat = np.asarray(self.intensity)[epoch, 1, individual, data, 0]
        else:
            int_mat = np.apply_over_axes(np.average,
                                         np.asarray(self.intensity)[epoch, 1, individual, data, 0],
                                         2)

        # for each individual
        for line in range(int_mat.shape[1]):
            # take the amount of epochs
            x = np.array([np.full(int_mat.shape[2], i) for i in range(int_mat.shape[0])]).flatten()
            # plot their intensity at each epoch
            # x = [0, 1, ... epoch n]
            # y = [intens of indv 'line' @ epoch 0, @ epoch 1, @ epoch 2, ... @ epoch n]
            y = int_mat[:, line].flatten()


            if linefmt == "scatter":
                plt.scatter(x, y, label = "individual %s" % line)
            else:
                plt.plot(x, y, label = "individual %s" % line)

        plt.xlabel("epoch")
        plt.ylabel("Intensity")

        plt.legend()

        plt.show()
        return None

    def animate(self):

        return None

class mirror_alg(genetic_algoritm):
    def __init__(self, dtype: str = "float", bitsize: int = 32,
                 endianness: str = "big"):
        super().__init__(dtype, bitsize, endianness)

        # self.log.__add__(log_intensity(self.b2n, self.bitsize, self.b2nkwargs))

    def run(self, selargs: dict = {},
            cargs: dict = {}, muargs: dict = {},
            target: float = 1, verbosity: int = 1):
        """
        :param cargs:
        :param muargs:
        :param seedargs:
        :param selargs:
        :param epochs:
        :param verbosity:
        :return:
        """
        global epoch, optimum, points_per_indv, points_stability_test

        if len(self.pop) == 0:
            self.init_pop()

        self.target = target

        selargs["fx"] = self.tfunc
        selargs["bitsize"] = self.bitsize
        selargs["b2n"] = self.b2n
        selargs["b2nkwargs"] = self.b2nkwargs
        selargs["verbosity"] = verbosity

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
                self.log.ranking.update(rank, fx, np.full(39, 0), 0)
                self.log.time.update(time() - self.tstart)
                self.log.selection.update(parents, p, fitness)
                # self.log.log_intensity.update(self.pop, intens.copy())

                if len(self.log.add_logs) > 0:
                    for l in self.log.add_logs:
                        l.update(data=self.pop)

                    self.log.sync_logs()

            elif self.dolog == 1:
                self.log.ranking.update(rank, fx, np.full(39, 0), 0)
                self.log.time.update(time() - self.tstart)

        while True:
            if np.average(intens[epoch][0, 0, 0, :]) < self.target:
                newgen = []
                if verbosity:
                    print("%s/%s" % (epoch + 1, self.epochs))

                cargs["bitsize"] = self.bitsize
                muargs["bitsize"] = self.bitsize

                for ppair in parents[self.elitism:]:
                    child1, child2 = self.cross(self.pop[ppair[0]], self.pop[ppair[1]], **cargs)

                    newgen.append(child1)
                    newgen.append(child2)


                for ppair in parents[:self.elitism]:
                    child1, child2 = self.cross(self.pop[ppair[0]], self.pop[ppair[1]], **cargs)
                    newgen.append(self.mutation(child1, **muargs))
                    newgen.append(self.mutation(child2, **muargs))



                # Select top10
                t10 = parents[:self.save_top]

                self.genlist.append([])

                for ppair in t10:
                    self.genlist[epoch].append(self.pop[ppair[0]])
                    self.genlist[epoch].append(self.pop[ppair[1]])

                self.genlist[epoch] = np.array(self.genlist[epoch])

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

                        self.log.ranking.update(rank, fx, np.full(39, 0), self.target)
                        self.log.time.update(time() - self.tstart)
                        self.log.selection.update(parents, p, fitness)
                        self.log.value.update(self.pop, self.genlist[epoch])
                        # self.log.log_intensity.update(self.pop, intens.copy())

                        # if additional logs added by appending them after initiation of self.log
                        # go through them and update with the population
                        # other data can be found within other logs and data
                        # can be added by using global statements or other.
                        if len(self.log.add_logs) > 0:
                            for l in self.log.add_logs:
                                l.update(data=self.pop)

                            self.log.sync_logs()


                    elif self.dolog == 1:
                        self.log.ranking.update(rank, fx, self.tfunc.minima["x"], self.tfunc.minima["fx"])
                        self.log.time.update(time() - self.tstart)

                print(np.average(intens[epoch][0, 0, 0, :]))

                epoch += 1
                self.results = self.genlist

                intens.append(intens_blueprint)
                individual_table.append(individual_table_blueprint)

            else:
                break
                set_voltage(individual_table[int(intens[epoch][0, 0, 2, 0])])


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




## Algorithm

from src.AdrianPackv402.Aplot import LivePlot
from time import time

k = 4
bitsize = 8
n = 39
size = [individuals, n]

ga = mirror_alg(bitsize=bitsize)

# for i in np.linspace(0.01, 0.1, 10):
try:
    def t():
        return time() - t0

    def main():

        ga.optimumfx = optimum
        print(size)
        ga.init_pop("nbit", shape=[size[0], size[1]], bitsize=bitsize)

        print(ga.log.creation)
        print(ga.pop.shape)
        ga.b2nkwargs = {"factor": 1, "normalised": True, "bitsize": 8, "bias": 0.0}

        ga.elitism = 4

        ga.b2n = ndbit2int

        ga.logdata(2)
        # ga.log.append(log_intensity(ga.b2n, ga,bitsize, ga.b2nkwargs))

        # ga.seed = uniform_bit_pop_float
        ga.set_cross(full_equal_prob)
        ga.set_mutate(full_mutate)
        ga.set_select(select)

        # P value for population of 20?
        p = 0.01
        ga.log.ranking.epoch.append(0)

        print("start run")
        ga.run(muargs={"mutate_coeff": 3},
               selargs={"nbit2num": ndbit2int,
                        "b2n": ga.b2n,
                        "b2nkwargs" : ga.b2nkwargs,
                        "p": p
                        },
               verbosity=1,
               target=4)

        # ga.log.log_intensity.plot(fmt_data = "raw", individual = slice(0, 1))
        # ga.save_log("dfmtest_data%s.pickle" % k)



    def checkruntime():
        sleep(runtime)
        return True


    if __name__ == "__main__":
        print("time: ", time() - t0)

        # with mp.Pool(2) as p:
        #     worker1 = p.apply_async(main, [])
        #     worker2 = p.apply_async(checkruntime, [])
        #
        #     worker2.wait()
        #     p.close()

    # main()
    # except:
    #     voltages = np.zeros(shape=n)
    #
    #     if not oko.set(handle, voltages):
    #         sys.exit("Error writing to OKODM device: " + oko.lasterror())
    #
    #     print("Voltages set to zero")

except Exception as e:
    print('\x1b[33m' + "Exception: %s " % e + '\x1b[0m')

finally:
    if __name__ == "__main__":
        k = 6
        k += 1

        ga.save_log("dfmfake_data%s.pickle" % k)

        # print(intens)

        set_voltage(np.zeros(shape=n))

        print("Mirror voltages set to zero")
        # print("Final best intensity: %s" % bestintens[-1, 0])
        print("Execution time: %s" % (time() - t0))
        print(r"Log saved to src\dfmtest_data%s.pickle" % k)
        print("Done")
        # sys.exit()

main()

time_intens = np.array(time_intens)
print(time_intens)
plt.plot(time_intens[:, 0], time_intens[:, 1])
plt.show()

