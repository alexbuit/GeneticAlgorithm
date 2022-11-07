import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from time import time

import numpy as np
from AdrianPack.Aplot import LivePlot, Default
from AdrianPack.Fileread import Fileread
from helper import Ndbit2float

from DFM_opt_alg import *

tstart = time()
tfunc = wheelers_ridge

low, high = 0, 2

gea = genetic_algoritm(bitsize=16)
gea.b2n = ndbit2int
# gea.load_log("wheeler100_noelite.pickle")
# print(gea.log)
# top = 1000
# pl = gea.log.fitness.plot(top=top, data_label="no elite")
#
# gea.load_log("wheeler100_bigelite.pickle")
# print(gea.log)
# pl2 = gea.log.fitness.plot(add_mode=True, colour="C1", top=top, data_label="big elite")
# plt.title("top: %s" % top)
# pl += pl2
#
# pl()

rng = np.logspace(-3, 1, 5)

figure = plt.figure()

fitness = []
pdict = {}
for i in range(rng.size):
    gea.load_log("wheeler16b_k%s.pickle" % i)
    print(gea.log)
    fitness.append(gea.log.fitness.data)
    pdict[i] = plt.plot(gea.log.fitness[0], label="k=%s" % rng[i], linestyle="-",
                        marker="")

plt.legend(loc="upper right")

def update(frame):
    plt.legend(loc="upper right")

    for i in range(rng.size):
        pdict[i][0].set_data(np.arange(0, 500), fitness[i][frame])

    plt.title("Iteration: %s" % frame)
    return None


animation = FuncAnimation(figure, update, interval=500,
                          frames=range(25))

plt.legend()
plt.xlabel("Rank")
plt.ylabel("normalised fitness")

animation.save("fitness_for_vallogk.gif", dpi=1000, writer=PillowWriter(fps=1))



# print(max(gea.log.value[0][:, 0][:, 1]))
# gea.log.fitness.plot(top=10)

# pl = gea.log.fitness.plot()
# print(pl)
# pl()

# gea.log.value.plot1d(tfunc, low, high, epoch=len(gea.log.value.epoch)-1)
# gea.log.value.animate1d(tfunc, low, high)

print("time: %s" % (time() - tstart))
