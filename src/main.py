import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from time import time

import numpy as np
from helper import Ndbit2float

from genetic_alg_mirror import log_intensity, select

from DFM_opt_alg import *

tstart = time()
low, high = 0, 5

gea = genetic_algoritm(bitsize=8)


# gea.load_log("Booth16b_p10.pickle", True)
gea.load_log("dfmtest_data7.pickle", True)
lg = gea.log

gea.b2n = lg.b2n
gea.b2nkwargs = lg.b2nkwargs


# lg.selection.plot(x_label="Iteration", y_label="avg Fitness", title="Fitness of individuals", fmt_data="raw", top=1)

print(lg.selection.data)

# print(lg.add_logs)
#
# print(lg.log_intensity)
# lg.log_intensity.plot(linefmt="plot")
# bestsl = np.array(lg.log_intensity.bestsol)[:, :, 0]

# print(bestsl)

# print(np.apply_over_axes(np.average, bestsl, 1))
# plt.plot(np.arange(0, 21), np.apply_over_axes(np.average, bestsl, 1).flatten())

# lg.selection
# plt.plot(np.arange(0, 21), np.apply_over_axes(np.max, np.array(lg.ranking.effectivity), 1).flatten())#
# plt.show()
# def inv_ackley(x):
#     return booths_function(x)

# print(39 * -39.16599)
# print([Styblinski_Tang(i[0]) for i in lg.ranking.bestsol])
# d = 39
# print([np.sqrt(sum((i[0] - np.full(d, -2.903534))**2)) for i in lg.ranking.bestsol])
#
# plt.plot(np.arange(0, 21), [np.sqrt(sum((i[0] - np.full(d, -2.903534))**2)) for i in lg.ranking.bestsol])
# # plt.plot(np.arange(0, 101), np.apply_over_axes(np.min, np.array(lg.ranking.distance), 1).flatten())#
# plt.show()

# lg.value.animate2d(michealewicz, -5, 5, fitness=lg.selection.fitness)
print("time: %s" % (time() - tstart))
