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
gea.load_log("wheeler16b_2.pickle")
print(gea.log)
# gea.log.value.animate2d(tfunc, low, high)

plt.plot(np.arange(0, 1000), gea.log.fitness[50], label = "k = 0.8")

gea.load_log("wheeler16b_3.pickle")
print(gea.log)

plt.plot(np.arange(0, 1000), gea.log.fitness[50], label="k = 2.5")
plt.legend()
plt.show()


# print(max(gea.log.value[0][:, 0][:, 1]))
# gea.log.fitness.plot(top=10)

# pl = gea.log.fitness.plot()
# print(pl)
# pl()

# gea.log.value.plot1d(tfunc, low, high, epoch=len(gea.log.value.epoch)-1)
# gea.log.value.animate1d(tfunc, low, high)

print("time: %s" % (time() - tstart))
