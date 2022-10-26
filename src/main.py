import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from time import time

import numpy as np
from AdrianPack.Aplot import LivePlot
from AdrianPack.Fileread import Fileread
from helper import Ndbit2float

from DFM_opt_alg import tfx, wheelers_ridge

tstart = time()
data = list(Fileread(path="GAmult_5_tfuncwheelers_ridge_bsize64_sim0.txt")().values())

low, high = -10, 10
popsize = [len(i) for i in data]

for i in range(len(data)):
    # print(data[i])
    # print(np.where(data[i] == "None"))
    # print(data[i])
    data[i] = np.delete(data[i], np.where(data[i] == "None"))

    resmat = np.empty((len(data[i]), len(data[i][0])), dtype=np.uint8)
    for x in range(len(data[i])):
        for b in range(len(data[i][x])):
            resmat[x, b] = int(data[i][x][b])

    data[i] = resmat

data_float = [Ndbit2float(i, 64) for i in data]
figure = plt.figure()

line, = plt.plot(data_float[0][:, 0], data_float[0][:, 1], linestyle="",
                                   marker="o", label="Algortithm")

x1, x2 = np.linspace(-2, 10, 1000), np.linspace(-10, 10, 1000)
X1, X2 = np.meshgrid(x1, x2)
y = wheelers_ridge([X1, X2])

plt.contour(X1, X2, y)

plt.xlim(min(x1), 10)
plt.ylim(-2.5, 5)

tx = plt.text(high - 3, 0, popsize[0])

def update(frame):
    global tx

    tx.remove()

    line.set_data(data_float[frame][:, 0], data_float[frame][:, 1])

    tx = plt.text(-5, tfx(high) - 40, "popsize: %s" % popsize[frame])

    plt.title("Iteration: %s" % frame)
    return None

animation = FuncAnimation(figure, update, interval=1000, frames=range(len(data)))
animation.save("wheelersridge_8.gif", dpi=300, writer=PillowWriter(fps=5))


print("time: %s" % (time() - tstart))

# popsize = [len(i) for i in data]
#
# figure = plt.figure()
#
# line, = plt.plot(data[0], tfx(data[0]), linestyle="",
#                                   marker="o", label="Algortithm")
#
# linsp = np.linspace(low, high, 1000)
# line_th = plt.plot(linsp, tfx(linsp), label="f($x$) = $3x^2 + 2x +1$")
#
# plt.xlim(min(linsp), max(linsp))
# plt.ylim(min(tfx(linsp)) - 10, max(tfx(linsp)))
#
# plt.xlabel("$x$")
# plt.ylabel("f($x$)")
#
# plt.grid()
# plt.legend(loc="upper right")
# tx = plt.text(high - 3, 0, popsize[0])
#
# def update(frame):
#     global tx
#
#     tx.remove()
#     data[frame] = data[frame][data[frame] < high]
#     data[frame] = data[frame][data[frame] > low]
#
#     line.set_data(data[frame], tfx(data[frame]))
#
#     tx = plt.text(-5, tfx(high) - 40, "popsize: %s" % popsize[frame])
#
#     plt.title("Iteration: %s" % frame)
#     return None
#
# animation = FuncAnimation(figure, update, interval=1000, frames=range(len(data) - 1))
# animation.save("smallpop2.gif", dpi=300, writer=PillowWriter(fps=5))