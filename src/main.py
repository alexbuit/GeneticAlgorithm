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

# lg.selection.plot(x_label="Individuals", y_label="Fitness", title="Fitness of individuals", fmt_data="raw")

lg.ranking.plot(top=2)

print(lg.ranking.bestsol)

def inv_ackley(x):
    return -ackley(x)

lg.time.plot()

lg.value.animate2d(inv_ackley, -5, 5)
print("time: %s" % (time() - tstart))
