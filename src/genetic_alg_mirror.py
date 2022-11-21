import random

import numpy as np
import matplotlib.pyplot as plt

from typing import Union
from time import sleep

from DFM_opt_alg import genetic_algoritm, full_mutate
from src.cross_funcs import full_single_point
from src.helper import ndbit2int
from src.selection_funcs import *
from src.log import log_object


## global variables
individuals: int = 20
points_per_indv: int = 100

intens: np.ndarray = np.zeros(shape=[individuals, points_per_indv])
optimum: float = 50e-6

epochs = 20  # ??
epoch = -1  # ??

if __name__ == "__main__":
    # # Opening the mirror handle
    # import okotech_lib.okodm_sdk.python.okodm_class as oko
    # import sys
    #
    # handle = oko.open("MMDM 39ch,15mm",
    #                   "USB DAC 40ch, 12bit")  # handle for other functions
    #
    # if handle == 0:
    #     sys.exit(("Error opening OKODM device: ", oko.lasterror()))

    # # Get the number of channels
    # n = oko.chan_n(handle) # Should be 39
    n = 39
    #
    # ## Opening the DMMM
    # import pyvisa
    # from ThorlabsPM100 import ThorlabsPM100
    #
    # rm = pyvisa.ResourceManager()
    # print(rm.list_resources())
    # inst = rm.open_resource('USB0::0x1313::0x8078::PM001464::INSTR', timeout=1)
    #
    # power_meter = ThorlabsPM100(inst=inst)

    def read_pm():
        return random.randint(0, 100)

## Additional functions for the algorithm
def select(*args, **kwargs):
    global intens, optimum, points_per_indv, epoch

    pop = args[0]

    b2n = kwargs["b2n"]
    b2nkwargs = kwargs["b2nkwargs"]

    # probability paramter for rank selection
    prob_param = 1.9
    if "p" in kwargs:
        prob_param = kwargs["p"]

    num_pop = b2n(pop, **b2nkwargs)

    # Values are normalised, how much time does this take?
    for indiv in num_pop:
        for var in indiv:
            try:
                assert -1 < var < 1
            except AssertionError:
                raise ValueError("Input voltage can not be > 1 or < -1")

    fitness = np.zeros(num_pop.shape[0])

    i = 0
    for indiv in num_pop:
        voltages = indiv

        # if not oko.set(handle, voltages):
        #     sys.exit("Error writing to OKODM device: " + oko.lasterror())


        for j in range(points_per_indv):
            intens[i, j] = read_pm()


        fitness[i] = np.average(intens[:, i])/optimum

        i += 1
        # Test out on DFM and append intensity
        # Divide intensity by global var optimum

    fit_rng = np.flip(np.argsort(fitness))

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

    voltages = np.zeros(shape=n)

    # if not oko.set(handle, voltages):
    #     sys.exit("Error writing to OKODM device: " + oko.lasterror())

    epoch += 1
    return pind

def fitness(*args, **kwargs):
    global intens, optimum, epoch

    if epoch is not None and epoch < epochs:
        intens_copy = np.average(intens[epoch])
        return intens_copy / optimum
    else:
        return np.zeros(shape=[individuals])

class log_intensity(log_object):
    def __init__(self, b2num, bitsize, b2nkwargs, *args, **kwargs):
        super().__init__(b2num, bitsize, b2nkwargs, *args, **kwargs)
        self.intensity = []

    def update(self, data, *args):
        global intens
        self.data.append(data)
        self.epoch.append(len(self.data))
        self.intensity.append(intens.copy())


    def __copy__(self):
        log_intens_c = log_intensity(self.b2n, self.bitsize, self.b2nkwargs)
        log_intens_c.intensity = self.intensity
        log_intens_c.data = self.data
        log_intens_c.epoch = self.epoch

        return log_intens_c

    def plot(self, epoch: Union[slice, int] = slice(0, None),
             individual: Union[slice, int] = slice(0, None),
             data: Union[slice, int] = slice(0, None),
             fmt_data: str = "average"):

        if fmt_data.lower() == "raw":
            int_mat = np.asarray(self.intensity)[epoch, individual, data]
        else:
            int_mat = np.apply_over_axes(np.average,
                                         np.asarray(self.intensity)[epoch, individual, data],
                                         2)

        for line in range(int_mat.shape[1]):
            print(int_mat.ndim)
            x = np.array([np.full(int_mat.shape[2], i) for i in range(int_mat.shape[0])]).flatten()
            y = int_mat[:, line].flatten()

            plt.scatter(x, y)



            print(x)
            print("----")
            print(y)

        plt.show()
        return None

    def animate(self):

        return None

## Algorithm
if __name__ == "__main__":
    bitsize = 8
    size = [individuals, n]

    ga = genetic_algoritm(bitsize=bitsize)

    print(ga.log.creation)
    ga.optimumfx = optimum
    ga.init_pop("nbit", shape=[size[0], size[1]], bitsize=bitsize)
    print(ga.pop.shape)
    ga.b2nkwargs = {"factor": 1, "normalised": True, "bitsize": 8}

    ga.elitism = 25

    ga.b2n = ndbit2int

    ga.logdata(2)
    ga.log.append(log_intensity(ga.b2n, ga,bitsize, ga.b2nkwargs))


    # ga.seed = uniform_bit_pop_float
    ga.set_cross(full_single_point)
    ga.set_mutate(full_mutate)
    ga.set_select(select)
    ga.fitness = fitness

    # P value for population of 20?
    p = 0.1
    ga.log.ranking.epoch.append(0)

    ga.run(epochs=epochs, muargs={"mutate_coeff": 2},
           selargs={"nbit2num": ndbit2int,
                    "b2n": ga.b2n,
                    "b2nkwargs" : ga.b2nkwargs,
                    "p": p
                    },
           verbosity=1)

    i = 1

    ga.log.log_intensity.plot(fmt_data = "raw", individual = slice(0, 1))
    ga.save_log("dfmtest_data%s.pickle" % i)