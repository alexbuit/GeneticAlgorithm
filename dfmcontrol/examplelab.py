import time
from typing import List, Callable

from dfmcontrol.AdrianPackv402.Helper import compress_ind
from dfmcontrol.DFM_opt_alg import genetic_algoritm
import dfmcontrol as dfmc
from dfmcontrol.test_functions import *
from dfmcontrol.test_functions import tfx_decorator

import dfmcontrol.helper as dfmh

import matplotlib.pyplot as plt
import numpy as np


# For the mirror

## global variables
individuals: int = 20
points_per_indv: int = 25
points_stability_test = 25

runtime = 10 # runtime in seconds
epoch = 0  # ??

# Hardcoded value for 39ch mirror
individual_size = 39

test_setup = True

bitsize = 8

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

t0 = time.time()


# Opening the mirror handle
import okotech_lib.okodm_sdk.python.okodm_class as oko
import sys

handle = oko.open("MMDM 39ch,15mm",
                  "USB DAC 40ch, 12bit")  # handle for other functions

if handle == 0:
    sys.exit(("Error opening OKODM device: ", oko.lasterror()))

# Get the number of channels
n = oko.chan_n(handle)  # Should be 39

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
    time_intens.append(np.array([power_meter.read, time.time() - t0]))
    return time_intens[-1][0]


def set_voltage(voltages):
    try:
        if not oko.set(handle, voltages):
            sys.exit("Error writing to OKODM device: " + oko.lasterror())
    except:
        voltages = np.zeros(shape=n)
        if not oko.set(handle, voltages):
            sys.exit("Error writing to OKODM device: " + oko.lasterror())

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

    ppi: int = kwargs.get("points_per_indv", 25)
    stability: bool = kwargs.get("stability", False)

    num_pop = b2n(pop, bitsize, **b2nkwargs)

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
        print(np.sum(voltages))
        set_voltage(voltages)

        # Read the power meter
        compress_this = np.zeros([2, ppi])
        # print(ppi, intens[0].shape)
        for j in range(ppi):
            if stability:
                compress_this[0, j] = read_pm()
                compress_this[1, j] = time.time() - t0
            else:
                intens[epoch][1, i, 0, j] = read_pm()
                intens[epoch][1, i, 1, j] = time.time() - t0
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
            avg_read[i] = np.average(intens[epoch][1, i, 0, :])

        i += 1
        # Test out on DFM and append intensity
        # Divide intensity by global var optimum

    set_voltage(np.zeros(shape=n))

    print(avg_read)

    return avg_read

# Create a genetic algorithm object
ga = genetic_algoritm(bitsize=8)
tfunc = tfx_decorator(tfmirror)
tfunc.set_minima(minima={"x": None, "fx": 40e-6})

# use the default bit2num function
ga.b2n = dfmh.ndbit2int

# The range of possible floats is now -5 to 5
ga.b2nkwargs = {"factor": 1}

# Initiate a population of 16 individuals with 2 genes each
ga.init_pop("normal", shape=[individuals, n], bitsize=ga.bitsize, factor=ga.b2nkwargs["factor"])
print(ga.b2n(ga.pop, ga.bitsize, **ga.b2nkwargs))
# Initiate the log
ga.logdata(2)

# Set the target function
ga.target_func(tfunc)

# Set the selection, mutation and crossover functions
ga.set_select(dfmc.selection_funcs.rank_selection) # Rank selection
ga.set_mutate(dfmc.mutation.full_mutate) # Mutate the full bit (do not use for IEEE 754 floats)
ga.set_cross(dfmc.cross_funcs.full_equal_prob) # Crossover the full bit (do not use for IEEE 754 floats)

ga.elitism = 8  # Keep the 4 best individuals
# Run the genetic algorithm

# The runcond argument is a string that is evaluated as a condition for the algorithm to stop
# The string can use any of the attributes of the genetic algorithm object
runcond = r"self.epoch < 50"  # Stop when the minimum distance from the best solution to the mathematical is less than 0.1

# The verbosity argument is an integer that controls the amount of output
# 0: No output
# 1: Output the current generation, (best & group) distance to the best solution and the best fitness
verbosity = 2

# The muargs argument is a dictionary that is passed to the mutation function as the kwargs argument
muargs = {"mutate_coeff": 3}

# The selargs argument is a dictionary that is passed to the selection function as the kwargs argument
selargs = {"nbit2num": ga.b2n, "fitness_func": dfmc.selection_funcs.no_fitness,
            "allow_duplicates": True, "avoid_calc_fx": True,
           "points_per_individual": points_per_indv}

# print(ga.b2n(ga.pop, ga.bitsize,**ga.b2nkwargs))

tstart = time.time()
set_voltage(np.zeros(n))


epochs = np.zeros([10, 3])

for i in range(5):
    # try:
    ga.run(muargs=muargs, selargs=selargs, verbosity=verbosity, runcond=runcond)
    print("iter %s, epochs %s" % (i, ga.epoch))

    print("Time elapsed: %s" % (time.time() - tstart))

    ga.save_log("Test_%s_%s.pickle" % (ga.select.__name__, i))
    np.savetxt("Test_%s_%s.txt" % (ga.select.__name__, i), time_intens)

    epochs[i, 0] = ga.epoch
    epochs[i, 1] = time.time() - tstart
    epochs[i, 2] = np.min(np.abs(ga.log.ranking.distancefx[-1]))
    ga.reset(reset_pop=False)
    set_voltage(np.zeros(n))

    time_intens: List[np.ndarray] = [np.zeros(2)]
    tstart = time.time()
    # except Exception as e:
    #     print(e)
    #     set_voltage(np.zeros(n))

np.savetxt("labtest1%s.csv" % ga.select.__name__, epochs, delimiter=";")


ga.set_select(dfmc.selection_funcs.roulette_selection)
epochs = np.zeros([10, 3])

for i in range(5):
    try:
        ga.run(muargs=muargs, selargs=selargs, verbosity=verbosity,
               runcond=runcond)
        print("iter %s, epochs %s" % (i, ga.epoch))

        print("Time elapsed: %s" % (time.time() - tstart))

        ga.save_log("Test_%s_%s.pickle" % (ga.select.__name__, i))
        np.savetxt("Test_%s_%s.txt" % (ga.select.__name__, i), time_intens)

        epochs[i, 0] = ga.epoch
        epochs[i, 1] = time.time() - tstart
        epochs[i, 2] = np.min(np.abs(ga.log.ranking.distancefx[-1]))
        ga.reset(reset_pop=False)
        set_voltage(np.zeros(n))

        time_intens: List[np.ndarray] = [np.zeros(2)]
        tstart = time.time()
    except Exception as e:
        print(e)
        set_voltage(np.zeros(n))

np.savetxt("labtest1%s.csv" % ga.select.__name__, epochs, delimiter=";")


ga.set_select(dfmc.selection_funcs.boltzmann_selection)
epochs = np.zeros([10, 3])

for i in range(5):
    try:
        ga.run(muargs=muargs, selargs=selargs, verbosity=verbosity,
               runcond=runcond)
        print("iter %s, epochs %s" % (i, ga.epoch))

        print("Time elapsed: %s" % (time.time() - tstart))

        ga.save_log("Test_%s_%s.pickle" % (ga.select.__name__, i))
        np.savetxt("Test_%s_%s.txt" % (ga.select.__name__, i), time_intens)

        epochs[i, 0] = ga.epoch
        epochs[i, 1] = time.time() - tstart
        epochs[i, 2] = np.min(np.abs(ga.log.ranking.distancefx[-1]))
        ga.reset(reset_pop=False)
        set_voltage(np.zeros(n))

        time_intens: List[np.ndarray] = [np.zeros(2)]
        tstart = time.time()
    except Exception as e:
        print(e)
        set_voltage(np.zeros(n))

np.savetxt("labtest1%s.csv" % ga.select.__name__, epochs, delimiter=";")

ga.set_select(dfmc.selection_funcs.rank_tournament_selection)
epochs = np.zeros([10, 3])


for i in range(5):
    try:
        ga.run(muargs=muargs, selargs=selargs, verbosity=verbosity,
               runcond=runcond)
        print("iter %s, epochs %s" % (i, ga.epoch))

        print("Time elapsed: %s" % (time.time() - tstart))

        ga.save_log("Test_%s_%s.pickle" % (ga.select.__name__, i))
        np.savetxt("Test_%s_%s.txt" % (ga.select.__name__, i), time_intens)

        epochs[i, 0] = ga.epoch
        epochs[i, 1] = time.time() - tstart
        epochs[i, 2] = np.min(np.abs(ga.log.ranking.distancefx[-1]))
        ga.reset(reset_pop=False)



        time_intens: List[np.ndarray] = [np.zeros(2)]
        tstart = time.time()
    except Exception as e:
        print(e)
        set_voltage(np.zeros(n))


np.savetxt("labtest1%s.csv" % ga.select.__name__, epochs, delimiter=";")

ga.set_select(dfmc.selection_funcs.rank_space_selection)
epochs = np.zeros([10, 3])

for i in range(5):
    try:
        ga.run(muargs=muargs, selargs=selargs, verbosity=verbosity,
               runcond=runcond)
        print("iter %s, epochs %s" % (i, ga.epoch))

        print("Time elapsed: %s" % (time.time() - tstart))

        ga.save_log("Test_%s_%s.pickle" % (ga.select.__name__, i))
        np.savetxt("Test_%s_%s.txt" % (ga.select.__name__, i), time_intens)

        epochs[i, 0] = ga.epoch
        epochs[i, 1] = time.time() - tstart
        epochs[i, 2] = np.min(np.abs(ga.log.ranking.distancefx[-1]))
        ga.reset(reset_pop=False)
        set_voltage(np.zeros(n))

        time_intens: List[np.ndarray] = [np.zeros(2)]
        tstart = time.time()
    except Exception as e:
        print(e)
        set_voltage(np.zeros(n))

np.savetxt("labtest1%s.csv" % ga.select.__name__, epochs, delimiter=";")

print("Time elapsed: %s" % (time.time() - tstart))

sys.exit()