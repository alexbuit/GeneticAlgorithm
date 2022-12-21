import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from time import time

import numpy as np
from helper import Ndbit2float

from genetic_alg_mirror import log_intensity, select

from DFM_opt_alg import *

tstart = time()
low, high = 0, 5

gea = genetic_algoritm(bitsize=16)


gea.load_log("Booth16b_p10.pickle", True)
lg = gea.log

gea.b2n = lg.b2n
gea.b2nkwargs = lg.b2nkwargs


# lg.ranking.plot()

# lg.selection.plot(x_label="Individuals", y_label="Fitness", title="Fitness of individuals", fmt_data="raw", top=5)

# lg.log_intensity.plot(linefmt="plot")
# bestsl = np.array(lg.log_intensity.bestsol)[:, :, 0]

# print(bestsl)

# print(np.apply_over_axes(np.average, bestsl, 1))
# plt.plot(np.arange(0, 21), np.apply_over_axes(np.average, bestsl, 1).flatten())

# lg.selection
# plt.plot(np.arange(0, 21), np.apply_over_axes(np.max, np.array(lg.ranking.effectivity), 1).flatten())#
#
#
# plt.show()
# def inv_ackley(x):
#     return booths_function(x)
#
# print(np.array(lg.selection.fitness))
lg.value.animate2d(michealewicz, -5, 5, fitness=lg.selection.fitness)
print("time: %s" % (time() - tstart))
