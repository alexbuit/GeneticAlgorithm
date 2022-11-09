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

low, high = 0, 5

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

gea.load_log("wheeler16b_m4.pickle")
ind = np.where(gea.log.ranking.effectivity[-1] == max(gea.log.ranking.effectivity[-1]))
print(max(gea.log.ranking.effectivity[-1]))
print(gea.log.ranking.ranknum[-1][ind])
pl0 = gea.log.ranking.plot(show=False, data_label="m=10", colour="C0", datasource=gea.log.ranking.distance,
                           x_label="Epoch", y_label="Effectivity")

gea.load_log("wheeler16b_m1.pickle")
pl1 = gea.log.ranking.plot(show=False, add_mode=True, data_label="m=2", colour="C1", datasource=gea.log.ranking.distance)

gea.load_log("wheeler16b_m2.pickle")
pl2 = gea.log.ranking.plot(show=False, add_mode=True, data_label="m=4", colour="C2", datasource=gea.log.ranking.distance)

gea.load_log("wheeler16b_m3.pickle")
pl3 = gea.log.ranking.plot(show=False, add_mode=True, data_label="m=6", colour="C3", datasource=gea.log.ranking.distance)

gea.load_log("wheeler16b_m0.pickle")
pl4 = gea.log.ranking.plot(show=False, add_mode=True, data_label="m=8", colour="C4", datasource=gea.log.ranking.distance)

pl0 += pl4
pl0 += pl1
pl0 += pl2
pl0 += pl3

plt.title("Effectivity of GA (pop=50) for different mutation rates")

pl0()

# rng = np.logspace(-3, 1, 5)
#
# figure = plt.figure()
#
# fitness = []
# pdict = {}
# for i in range(rng.size):
#     gea.load_log("wheeler16b_k%s.pickle" % i)
#     print(gea.log)
#     fitness.append(gea.log.fitness.data)
#     pdict[i] = plt.plot(gea.log.fitness[0], label="k=%s" % rng[i], linestyle="-",
#                         marker="")
#
# plt.legend(loc="upper right")
#
# def update(frame):
#     plt.legend(loc="upper right")
#
#     for i in range(rng.size):
#         pdict[i][0].set_data(np.arange(0, 500), fitness[i][frame])
#
#     plt.title("Iteration: %s" % frame)
#     return None
#
#
# animation = FuncAnimation(figure, update, interval=500,
#                           frames=range(25))
#
# plt.legend()
# plt.xlabel("Rank")
# plt.ylabel("normalised fitness")
#
# plt.show()


# print(max(gea.log.value[0][:, 0][:, 1]))
# gea.log.fitness.plot(top=10)

# pl = gea.log.fitness.plot()
# print(pl)
# pl()

# gea.log.value.plot1d(tfunc, low, high, epoch=len(gea.log.value.epoch)-1)
# gea.log.value.animate1d(tfunc, low, high)

print("time: %s" % (time() - tstart))
