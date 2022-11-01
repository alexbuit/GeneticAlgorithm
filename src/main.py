import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from time import time

import numpy as np
from AdrianPack.Aplot import LivePlot, Default
from AdrianPack.Fileread import Fileread
from helper import Ndbit2float

from DFM_opt_alg import *

tstart = time()
data = list(Fileread(path="ackley2_1.txt", dtype="str", delimiter=";")().values())
# print(Fileread(path="ackley1.txt", dtype="str", delimiter=";")())
#
low, high = -1, 1
popsize = [len(i) for i in data]

# for i in range(len(data)):
#     # print(data[i])
#     # print(np.where(data[i] == "None"))
#     # print(data[i])
#     data[i] = np.delete(data[i], np.where(data[i] == "None"))
#     print(data[i])
#     print((len(data[i]), len(data[i][0])))
#     resmat = np.empty((len(data[i]), len(data[i][0])), dtype=np.uint8)
#     for x in range(len(data[i])):
#         for b in range(len(data[i][x])):
#             resmat[x, b] = int(data[i][x][b])
#
#     data[i] = resmat
#
# data_float = [ndbit2int(i, 8) for i in data]

# print(np.loadtxt("ackley1.txt", delimiter=";"))

ga = genetic_algoritm()
ga.load_results("ackley4inv.txt")
data_float = ga.get_numeric(bit2num=ndbit2int, bitsize=9)

popsize = [len(i) for i in data_float]

figure = plt.figure()

line, = plt.plot(data_float[0][:, 0], data_float[0][:, 1], linestyle="",
                                   marker="o", label="Algortithm")

x1, x2 = np.linspace(low, high, 1000), np.linspace(low, high, 1000)
X1, X2 = np.meshgrid(x1, x2)
y = ackley([X1, X2])

plt.pcolormesh(X1, X2, y, cmap='RdBu', shading="auto")

plt.xlim(low, high)
plt.ylim(low, high)

plt.legend(loc="upper right")
plt.colorbar()

tx = plt.text(high - 3, 0, popsize[0])

def update(frame):
    global tx

    print(frame)

    tx.remove()

    line.set_data(data_float[frame][:, 0], data_float[frame][:, 1])

    tx = plt.text(-5, tfx(high) - 40, "popsize: %s" % popsize[frame])

    plt.title("Iteration: %s" % frame)
    return None

animation = FuncAnimation(figure, update, interval=500, frames=range(len(data_float)))
animation.save("ackley4inv.gif", dpi=600, writer=PillowWriter(fps=1))

# plt.show()

# def ologn(n, a, b, c):
#     return a * np.log2(n * b) + c
#
# # N-epochs = 25
# data = Fileread("gatimetestepoch32.txt")()
# data2 = Fileread("gatimetestepoch64.txt")()
# print(data)
# pl = Default(data["0"], data["1"]/10, linestyle=" ", degree=1, data_label="bitsize 32", colour="C0", x_label = "Time", y_label="pop size")
# pl2 = Default(data2["0"], data2["1"]/10, add_mode=True, linestyle=" ", data_label="bitsize 64", colour="C1", fx=ologn)
#
# pl += pl2
# pl()

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