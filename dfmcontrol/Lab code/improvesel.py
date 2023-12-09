
import time

from dfmcontrol.Optimisation import genetic_algoritm
import dfmcontrol as dfmc
from dfmcontrol.Mathematical_functions import *

import dfmcontrol.Helper as dfmh

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

# Create a genetic algorithm object
ga = genetic_algoritm(bitsize=16)
tfunc = Styblinski_Tang

# use the default bit2num function
ga.b2n = dfmh.ndbit2int

# The range of possible floats is now -5 to 5
ga.b2nkwargs = {"factor": 5}

# Initiate a population of 16 individuals with 2 genes each
ga.init_pop("normal", shape=[50, 39], bitsize=ga.bitsize, factor=ga.b2nkwargs["factor"])

# Initiate the log
ga.logdata(2)

# Set the target function
ga.target_func(tfunc)

# Set the selection, mutation and crossover functions
ga.set_select(dfmc.selection_funcs.roulette_selection) # Rank selection
ga.set_mutate(dfmc.mutation.mutate) # Mutate the full bit (do not use for IEEE 754 floats)
ga.set_cross(dfmc.cross_funcs.equal_prob) # Crossover the full bit (do not use for IEEE 754 floats)

ga.elitism = 8  # Keep the 8 best individuals
# Run the genetic algorithm

# The runcond argument is a string that is evaluated as a condition for the algorithm to stop
# The string can use any of the attributes of the genetic algorithm object
runcond = r"self.epoch < 150"  # Stop when the minimum distance from the best solution to the mathematical is less than 0.1

# The verbosity argument is an integer that controls the amount of output
# 0: No output
# 1: Output the current generation, (best & group) distance to the best solution and the best fitness
verbosity = 0

# The muargs argument is a dictionary that is passed to the mutation function as the kwargs argument
muargs = {"mutate_coeff": 3}

# The selargs argument is a dictionary that is passed to the selection function as the kwargs argument
selargs = {"nbit2num": ga.b2n, "fitness_func": dfmc.selection_funcs.no_fitness,
            "allow_duplicates": True}

# print(ga.b2n(ga.pop, ga.bitsize,**ga.b2nkwargs))
epochs = np.zeros([100, 4])

tstart = time.time()

tfuncs = [ackley, Styblinski_Tang, michealewicz]
tfuncsstr = ["ackley", "Styblinski_Tang", "michealewicz" ]
selfuncs = [dfmc.selection_funcs.roulette_selection]

ga.target_func(tfuncs[0])
# ga.run(muargs=muargs, selargs=selargs, verbosity=1, runcond=runcond)
#
# log = ga.log
#
# plt.plot(np.arange(len(log.ranking.result)), [np.average(i) for i in log.ranking.result], label="Group average")
# plt.plot(np.arange(len(log.ranking.distancefx)), [np.min(i) for i in log.ranking.distancefx], label="Best result")
#
# plt.grid()
#
# plt.xlabel("Iterations $i_{iter}$ [-]")
# plt.ylabel("Distance to solution $\Delta f(x_1, x_2)$ [-]")
#
# plt.title("Test function: 39-d Ackley function with roulette selection")
#
# plt.legend()
#
# plt.show()

for k in range(3):
    ga.target_func(tfuncs[ k])
    for j in range(5):
        epochs = np.zeros([100, 4])

        ga.set_select(selfuncs[j])
        for i in range(100):

            ga.run(muargs=muargs, selargs=selargs, verbosity=verbosity, runcond=runcond)

            print("iter %s, epochs %s func %s" % (i, ga.epoch, tfuncsstr[ k]))
            print("Time elapsed: %s" % (time.time() - tstart))

            yvalsmin = np.array([np.min(l) for l in ga.log.ranking.result])

            epochs[i, 0] = ga.epoch
            epochs[i, 1] = time.time() - tstart
            epochs[i, 2] = np.min(np.abs(ga.log.ranking.distancefx[-1]))
            epochs[i, 3] = (np.min(yvalsmin) - np.average(ga.log.ranking.result[0])) / np.average(ga.log.ranking.result[0]) * 100
            print((np.min(yvalsmin) - np.average(ga.log.ranking.result[0])) / np.average(ga.log.ranking.result[0]) * 100)
            print(np.average(ga.log.ranking.result[0]))
            print(np.min(yvalsmin))
            print(np.min(ga.log.ranking.distancefx[-1]))
            ga.reset(reset_pop=False)
            tstart = time.time()

        np.savetxt("Improveseltest%s_%s.csv" % (tfuncsstr[k], selfuncs[j].__name__), epochs, delimiter=";")
        mu, std = norm.fit(epochs[:, 3])
        print(mu, std)

print("Time elapsed: %s" % (time.time() - tstart))