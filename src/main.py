import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
from AdrianPack.Aplot import LivePlot
from AdrianPack.Fileread import Fileread

from DFM_opt_alg import tfx

low, high = -5, 5

data = list(Fileread(path="tdataGA_6.txt")().values())

popsize = [len(i) for i in data]

figure = plt.figure()

line, = plt.plot(data[0], tfx(data[0]), linestyle="",
                                  marker="o", label="Algortithm")

linsp = np.linspace(low, high, 1000)
line_th = plt.plot(linsp, tfx(linsp), label="f($x$) = $3x^2 + 2x +1$")

plt.xlim(min(linsp), max(linsp))
plt.ylim(min(tfx(linsp)) - 10, max(tfx(linsp)))

plt.xlabel("$x$")
plt.ylabel("f($x$)")

plt.grid()
plt.legend(loc="upper right")
tx = plt.text(high - 3, 0, popsize[0])

def update(frame):
    global tx

    tx.remove()
    data[frame] = data[frame][data[frame] < high]
    data[frame] = data[frame][data[frame] > low]

    line.set_data(data[frame], tfx(data[frame]))

    tx = plt.text(-5, tfx(high) - 40, "popsize: %s" % popsize[frame])

    plt.title("Iteration: %s" % frame)
    return None

animation = FuncAnimation(figure, update, interval=1000, frames=range(len(data) - 1))
animation.save("smallpop2.gif", dpi=300, writer=PillowWriter(fps=5))