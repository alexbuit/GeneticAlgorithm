import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from time import time

import numpy as np
from AdrianPack.Aplot import LivePlot, Default
from AdrianPack.Fileread import Fileread
from helper import Ndbit2float

from genetic_alg_mirror import log_intens, select, fitness

from DFM_opt_alg import *

tstart = time()
low, high = 0, 5

gea = genetic_algoritm(bitsize=8)
gea.b2n = ndbit2int

gea.load_log("dfmtest_nrnd1.pickle", False)



intens = np.apply_over_axes(np.average, gea.log.add_logs[0].intensity[-1], 1)
intens1 = np.apply_over_axes(np.average, gea.log.add_logs[0].intensity[0], 1)

plt.plot(intens, linestyle="--", marker="o")
plt.plot(intens1, linestyle="--", marker="o")
plt.show()

print("time: %s" % (time() - tstart))
