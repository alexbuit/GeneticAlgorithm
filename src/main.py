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

gea = genetic_algoritm(bitsize=9)
gea.b2n = ndbit2int

gea.load_log("Booth16b_p10.pickle", True)
lg = gea.log
print(lg.ranking.bestsol)
plt.scatter(np.linspace(0, 250, 250), lg.fitness[4])
plt.show()
# lg.value.animate2d(booths_function, -50, 50)
print("time: %s" % (time() - tstart))
