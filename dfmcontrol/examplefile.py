import time

from dfmcontrol.DFM_opt_alg import genetic_algoritm
import dfmcontrol as dfmc
from dfmcontrol.test_functions import *

import dfmcontrol.helper as dfmh

import matplotlib.pyplot as plt
import numpy as np

# Create a genetic algorithm object
ga = genetic_algoritm(bitsize=16)
tfunc = Styblinski_Tang

# use the default bit2num function
ga.b2n = dfmh.ndbit2int

# The range of possible floats is now -5 to 5
ga.b2nkwargs = {"factor": 5}

# Initiate a population of 16 individuals with 2 genes each
ga.init_pop("normal", shape=[50, 2], bitsize=ga.bitsize, factor=ga.b2nkwargs["factor"])

# Initiate the log
ga.logdata(2)

# Set the target function
ga.target_func(tfunc)

# Set the selection, mutation and crossover functions
ga.set_select(dfmc.selection_funcs.boltzmann_selection) # Rank selection
ga.set_mutate(dfmc.mutation.full_mutate) # Mutate the full bit (do not use for IEEE 754 floats)
ga.set_cross(dfmc.cross_funcs.full_equal_prob) # Crossover the full bit (do not use for IEEE 754 floats)

ga.elitism = 4  # Keep the 4 best individuals
# Run the genetic algorithm

# The runcond argument is a string that is evaluated as a condition for the algorithm to stop
# The string can use any of the attributes of the genetic algorithm object
runcond = r"np.min(np.abs(self.log.ranking.distancefx[-1])) > .01"  # Stop when the minimum distance from the best solution to the mathematical is less than 0.1

# The verbosity argument is an integer that controls the amount of output
# 0: No output
# 1: Output the current generation, (best & group) distance to the best solution and the best fitness
verbosity = 1

# The muargs argument is a dictionary that is passed to the mutation function as the kwargs argument
muargs = {"mutate_coeff": 3}

# The selargs argument is a dictionary that is passed to the selection function as the kwargs argument
selargs = {"nbit2num": ga.b2n, "fitness_func": dfmc.selection_funcs.no_fitness,
            "allow_duplicates": True}

# print(ga.b2n(ga.pop, ga.bitsize,**ga.b2nkwargs))
epochs = np.zeros([100, 3])

tstart = time.time()

for i in range(100):
    ga.run(muargs=muargs, selargs=selargs, verbosity=verbosity, runcond=runcond)
    print("iter %s, epochs %s" % (i, ga.epoch))

    print("Time elapsed: %s" % (time.time() - tstart))

    epochs[i, 0] = ga.epoch
    epochs[i, 1] = time.time() - tstart
    epochs[i, 2] = np.min(np.abs(ga.log.ranking.distancefx[-1]))
    ga.reset(reset_pop=False)
    tstart = time.time()

np.savetxt("epochs6m2.csv", epochs, delimiter=";")

plt.hist(epochs[:, 0], 30)
plt.xlabel("Epochs")
plt.ylabel("Frequency")

plt.show()

print("Time elapsed: %s" % (time.time() - tstart))

# plt.plot(np.arange(len(log.selection.fitness)), [np.average(i) for i in log.selection.fitness], label="Group average")
# plt.plot(np.arange(len(log.selection.fitness)), [np.min(i) for i in log.selection.fitness], label="Best result")
#
# plt.xlabel("Generation")
# plt.ylabel("Distance to the best solution")
#
# plt.title("Result of the genetic algorithm")
#
# plt.legend()
#
# plt.show()