import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from time import time

import numpy as np
from AdrianPack.Aplot import LivePlot, Default
from AdrianPack.Fileread import Fileread
from helper import Ndbit2float

from genetic_alg_mirror import log_intensity, select, fitness

from DFM_opt_alg import *

tstart = time()
low, high = 0, 5

gea = genetic_algoritm(bitsize=32)


gea.load_log("Booth16b_p10.pickle", True)
lg = gea.log

gea.b2n = lg.b2n
gea.b2nkwargs = lg.b2nkwargs

print(lg.ranking.bestsol)
print(lg.selection.fitness[-1].size)

# lg.ranking.plot()
#

print(lg.value.numvalue[0])
print(np.apply_along_axis(ackley, 1, lg.value.numvalue[0]))



print([np.min(np.apply_along_axis(ackley, 1, lg.value.numvalue[i])) for i in range(10)])

lg.value.animate2d(ackley, -4, 4, save_as="michealewicz.gif")
print("time: %s" % (time() - tstart))
