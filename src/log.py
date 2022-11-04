from typing import Callable
from datetime import datetime
from helper import convertpop2n
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from AdrianPack.Aplot import Default


class log:

    def __init__(self, pop, select, cross, mutate, fitness, b2n, elitism, savetop,
                 bitsize: int):
        # Pop
        self.pop = pop
        self.bitsize = bitsize

        # Logs
        self.time = log_time(b2n, bitsize)
        self.ranking = log_ranking(b2n, bitsize)
        self.fitness = log_fitness(b2n, bitsize)
        self.value = log_value(b2n, bitsize)

        # Used methods
        self.select: Callable = select
        self.cross: Callable = cross
        self.mutation: Callable = mutate
        self.fitnessfunc: Callable = fitness

        # Used variables
        self.elitism = elitism
        self.savetop = savetop

        # Binary to numerical conversion
        self.b2n = b2n

        self.logdict = {}

        self.creation = datetime.now()

    def __getitem__(self, item):
        return self.logdict[item]

    def __repr__(self):
        return ""

    def __str__(self):
        return "Log for GA created on %s" % self.creation

    def __copy__(self):
        logcopy = log(self.pop, self.select, self.cross, self.mutation,
                       self.fitness, self.b2n, self.elitism, self.savetop,
                       self.bitsize)

        logcopy.creation = self.creation

        logcopy.time.data = self.time.data
        logcopy.time.epoch = self.time.epoch

        logcopy.ranking.data = self.ranking.data
        logcopy.ranking.epoch = self.ranking.epoch
        logcopy.ranking.ranknum = self.ranking.ranknum


        logcopy.fitness.data = self.fitness.data
        logcopy.fitness.epoch = self.fitness.epoch

        logcopy.value.data = self.value.data
        logcopy.value.epoch = self.value.epoch
        logcopy.value.value = self.value.value
        logcopy.value.numvalue = self.value.numvalue
        logcopy.value.topx = self.value.topx
        return logcopy

    def copy(self):
        return self.__copy__()


class log_object:

    def __init__(self, b2num, bitsize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = []
        self.data = []

        self.bitsize = bitsize
        self.b2n = b2num

        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, item: int):
        return self.data[item]

    def __repr__(self):
        return str(self.data)

    def __copy__(self):
        object_copy = log_object(self.b2n, self.bitsize, *self.args, **self.kwargs)
        object_copy.data = self.data
        object_copy.epoch = self.epoch
        return object_copy

    def copy(self):
        return self.__copy__()

    def savetxt(self, path):

        return None

    def update(self, data, *args):
        self.data.append(data)
        self.epoch.append(len(self.data))
        return None



class log_time(log_object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __copy__(self):
        object_copy = log_time(self.b2n, self.bitsize, *self.args, **self.kwargs)
        object_copy.data = self.data
        object_copy.epoch = self.epoch
        return object_copy

    def plot(self):
        pass


class log_ranking(log_object):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranknum = []

    def __copy__(self):
        object_copy = log_ranking(self.b2n, self.bitsize, *self.args, **self.kwargs)
        object_copy.data = self.data
        object_copy.epoch = self.epoch
        object_copy.ranknum = self.ranknum
        return object_copy

    def update(self, data, *args):
        self.data.append(data)
        self.epoch.append(len(self.data))
        self.ranknum.append(self.b2n(data, self.bitsize))

    def __getitem__(self, item: int):
        return self.ranknum[item]

    def __repr__(self):
        return str(self.ranknum)


class log_fitness(log_object):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def __copy__(self):
        object_copy = log_fitness(self.b2n, self.bitsize, *self.args, **self.kwargs)
        object_copy.data = self.data
        object_copy.epoch = self.epoch
        return object_copy

    def plot(self, top: int = None, show: bool = True, save_as: str = "",**kwargs):
        if top == None:
            top = len(self.data[0])

        avgpepoch = [np.average(i[:top]) for i in self.data]

        # plt.plot(self.epoch, avgpepoch, linestyle="-", marker="o", label="Fitness")
        #
        # plt.ylabel("Fitness coefficient")
        # plt.xlabel("Epochs")
        #
        # plt.grid()
        #
        # plt.legend()
        # plt.show()

        pl = Default(self.epoch, avgpepoch, linestyle="-", marker="o", **kwargs)
        if show:
            if save_as != "":
                pl.save_as = save_as

            pl()


        return pl



class log_value(log_object):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = []
        self.numvalue = []
        self.topx = []

    def __getitem__(self, item):
        return self.numvalue[item]

    def __repr__(self):
        return str(self.value)

    def __copy__(self):
        object_copy = log_value(self.b2n, self.bitsize, *self.args, **self.kwargs)
        object_copy.data = self.data
        object_copy.epoch = self.epoch
        object_copy.value = self.value
        object_copy.numvalue = self.numvalue
        object_copy.topx = self.topx
        return object_copy

    def update(self, data, *args):
        self.value.append(data)
        self.epoch.append(len(self.data))
        self.numvalue.append(np.asarray(convertpop2n(bit2num=self.b2n,
                                                     target=list(data),
                                                     bitsize=self.bitsize)))
        self.topx.append(args[0])
        return None

    def topvalue(self, item = None):
        """

        :param item:
        :return:
        """
        if item == None:
            return self.topx
        else:
            return self.topx[item]

    def numerical(self, item = None):
        """

        :param item:
        :return:
        """
        if item == None:
            return self.numvalue
        else:
            return self.numvalue[item]

    def plot1d(self, tfx, low, high, save_as="", epoch=0):
        plt.figure()

        plt.plot(self.numvalue[epoch][:, 0], tfx(self.numvalue[epoch][:, 0]),
                 linestyle="",
                 marker="o", label="Algortithm")

        linsp = np.linspace(low, high)
        plt.plot(linsp, tfx(linsp), label="tfx", linestyle="--")

        plt.xlim(low, high)
        plt.ylim(low, high)

        plt.legend(loc="upper right")
        plt.colorbar()

        plt.tight_layout()

        if save_as == "":
            plt.show()
            return None

        plt.savefig(save_as, dpi=600)

        return None

    def plot2d(self, tfx, low, high, save_as="", epoch=0):
        """

        :param tfx:
        :param low:
        :param high:
        :param save_as:
        :param epoch:
        :return:
        """
        plt.figure()

        plt.plot(self.numvalue[epoch][:, 0][:, 0], self.numvalue[epoch][:, 0][:, 1],
                         linestyle="",
                         marker="o", label="Algortithm")

        x1, x2 = np.linspace(low, high, 1000), np.linspace(low, high, 1000)
        X1, X2 = np.meshgrid(x1, x2)
        y = tfx([X1, X2])

        plt.pcolormesh(X1, X2, y, cmap='RdBu', shading="auto")

        plt.xlim(low, high)
        plt.ylim(low, high)

        plt.legend(loc="upper right")
        plt.colorbar()

        plt.tight_layout()

        if save_as == "":
            plt.show()
            return None

        plt.savefig(save_as, dpi=600)

        return None

    # Save a gif of all epochs
    def animate1d(self, tfx: Callable, low: int, high: int, save_as=""):
        """

        :param tfx:
        :param low:
        :param high:
        :param save_as:
        :return:
        """

        figure = plt.figure()

        line, = plt.plot(self.numvalue[0][:, 0], tfx(self.numvalue[0][:, 0]),
                         linestyle="",
                         marker="o", label="Algortithm")

        linsp = np.linspace(low, high)
        plt.plot(linsp, tfx(linsp), label="tfx", linestyle="--")

        plt.xlim(low, high)
        plt.ylim(low, high)

        plt.legend(loc="upper right")

        def update(frame):
            print(frame)
            line.set_data(self.numvalue[frame][:, 0], tfx(self.numvalue[frame][:, 0]))

            plt.title("Iteration: %s" % frame)
            return None

        animation = FuncAnimation(figure, update, interval=500,
                                  frames=range(len(self.epoch)))

        if save_as == "":
            plt.show()
            return None

        animation.save(save_as, dpi=600, writer=PillowWriter(fps=1))

        return None

    def animate2d(self, tfx: Callable, low: int, high: int, save_as=""):
        """

        :param tfx:
        :param low:
        :param high:
        :param save_as:
        :return:
        """

        figure = plt.figure()

        line, = plt.plot(self.numvalue[0][:, 0][:, 0], self.numvalue[0][:, 0][:, 1],
                         linestyle="",
                         marker="o", label="Algortithm")

        x1, x2 = np.linspace(low, high, 1000), np.linspace(low, high, 1000)
        X1, X2 = np.meshgrid(x1, x2)
        y = tfx([X1, X2])

        plt.pcolormesh(X1, X2, y, cmap='RdBu', shading="auto")

        plt.xlim(low, high)
        plt.ylim(low, high)

        plt.legend(loc="upper right")
        plt.colorbar()

        def update(frame):
            print(frame)
            line.set_data(self.numvalue[frame][:, 0][:, 0], self.numvalue[frame][:, 0][:, 1])

            plt.title("Iteration: %s" % frame)
            return None

        animation = FuncAnimation(figure, update, interval=500,
                                  frames=range(len(self.epoch)))

        if save_as == "":
            plt.show()
            return None

        animation.save(save_as, dpi=600, writer=PillowWriter(fps=1))

        return None


