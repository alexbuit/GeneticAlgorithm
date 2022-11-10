import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from time import time

import numpy as np
from AdrianPack.Aplot import LivePlot, Default
from AdrianPack.Fileread import Fileread
from helper import Ndbit2float

from DFM_opt_alg import *

tstart = time()
tfunc = michealewicz

low, high = 0, 5

gea = genetic_algoritm(bitsize=32)
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

# gea.load_log("Micheal16b_m10.pickle")
# pl0 = gea.log.ranking.plot(show=False, data_label="m=1", colour="C0", datasource=gea.log.ranking.distance,
#                            x_label="Epoch", y_label="Effectivity", linestyle="-")
# print(gea.log.ranking.bestsol[-1])

gea.load_log("Micheal16b_p4.pickle")
pl0 = gea.log.ranking.plot(datasource=gea.log.ranking.distance, show=False, data_label="k = 10", colour="C0")
print(gea.log.ranking.bestsol[-1])
gea.load_log("Micheal16b_p0.pickle")
pl1 = gea.log.ranking.plot(datasource=gea.log.ranking.distance, show=False, data_label="k = 2", add_mode=True, colour="C1")
print(gea.log.ranking.bestsol[-1])
gea.load_log("Micheal16b_p1.pickle")
pl2 = gea.log.ranking.plot(datasource=gea.log.ranking.distance, show=False, data_label="k = 4", add_mode=True, colour="C2")
print(gea.log.ranking.bestsol[-1])
gea.load_log("Micheal16b_p2.pickle")
pl3 = gea.log.ranking.plot(datasource=gea.log.ranking.distance, show=False, data_label="k = 6", add_mode=True, colour="C3")
print(gea.log.ranking.bestsol[-1])
gea.load_log("Micheal16b_p3.pickle")
pl4 = gea.log.ranking.plot(datasource=gea.log.ranking.distance, show=False, data_label="k = 8", add_mode=True, colour="C4")
print(gea.log.ranking.bestsol[-1])
pl0 += pl1
pl0 += pl2
pl0 += pl3
pl0 += pl4

pl0()

# plt.plot(gea.log.fitness[0])
# plt.show()

gea.log.value.animate2d(michealewicz, -10, 10)


print("time: %s" % (time() - tstart))
