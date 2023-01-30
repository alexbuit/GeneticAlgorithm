from typing import Callable
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter



try:
    from .helper import convertpop2n
    from .AdrianPackv402.Aplot import Default
except ImportError:
    from dfmcontrol.helper import convertpop2n
    from dfmcontrol.AdrianPackv402.Aplot import Default

def pythagoras(x):
    return np.sqrt(sum(x))


class log:
    """
    Log object containing all data collected by the GA during the run.

    The methods are usually called by the GA itself, but can be called manually
    before the GA is run to add a log object to the log.

    Methods
    -------
    add_log(log_object)
        Add a log object to the log.
    sync_logs()
        Sync all log objects in the log.

    Attributes
    ----------

    Attributes containing the log files with data collected by the GA during the run.
    --

    time
        Log object containing the time data.
    ranking
        Log object containing the ranking data.
    selection
        Log object containing the selection data.
    value
        Log object containing the value data.
    add_logs
        List of log objects added to the log.

    Attributes regarding the settings of the GA.
    --

    pop
        The population used in the GA.
    bitsize
        The size of the binary representation of the population.
    select
        The selection method used in the GA.
    cross
        The crossover method used in the GA.
    mutation
        The mutation method used in the GA.
    elitism
        The elitism used in the GA.
    savetop
        The savetop used in the GA.
    b2n
        The binary to numerical conversion method used in the GA.
    b2nkwargs
        The keyword arguments used in the binary to numerical conversion method.
    logdict
        Dictionary containing all log objects.
    creation
        The datetime of the creation of the log.
    """
    def __init__(self, pop, select, cross, mutate, b2n, elitism, savetop,
                 bitsize: int, b2nkwargs: dict):
        # Pop
        self.pop = pop
        self.bitsize = bitsize

        # Logs
        self.time = log_time(b2n, bitsize, b2nkwargs)
        self.ranking = log_ranking(b2n, bitsize, b2nkwargs)
        self.selection = log_selection(b2n, bitsize, b2nkwargs)
        self.value = log_value(b2n, bitsize, b2nkwargs)

        self.add_logs = []

        # Used methods
        self.select: Callable = select
        self.cross: Callable = cross
        self.mutation: Callable = mutate

        # Used variables
        self.elitism = elitism
        self.savetop = savetop

        # Binary to numerical conversion
        self.b2n = b2n
        self.b2nkwargs = b2nkwargs

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
                       self.b2n, self.elitism, self.savetop,
                       self.bitsize, self.b2nkwargs)

        logcopy.creation = self.creation

        logcopy.time.data = self.time.data
        logcopy.time.epoch = self.time.epoch

        logcopy.ranking.data = self.ranking.data
        logcopy.ranking.epoch = self.ranking.epoch
        logcopy.ranking.ranknum = self.ranking.ranknum
        logcopy.ranking.effectivity = self.ranking.effectivity
        logcopy.ranking.distancex = self.ranking.distancex
        logcopy.ranking.distancefx = self.ranking.distancefx
        logcopy.ranking.bestsol = self.ranking.bestsol

        logcopy.value.data = self.value.data
        logcopy.value.epoch = self.value.epoch
        logcopy.value.value = self.value.value
        logcopy.value.numvalue = self.value.numvalue
        logcopy.value.topx = self.value.topx
        return logcopy

    def copy(self):
        """ Return a copy of the log. """
        return self.__copy__()

    def __add__(self, other: "log_object"):
        setattr(self, other.__class__.__name__, other)
        print(other)
        print("add:", getattr(self, other.__class__.__name__)) # hier gaat fout
        self.add_logs.append(other)
        return self.__copy__()

    def append(self, other):
        """ Append a log object to the log. """
        return self.__add__(other)

    def sync_logs(self):
        """ Sync all log objects in the log. """
        for lg in self.add_logs:
            setattr(self, lg.__class__.__name__, lg.copy())

        return None

class log_object:
    """
    A template for a log object.

    This template can be used to create a log object that can be added to the log using log.append(),
    log += or log.__add__().

    The methods can be customized to fit the needs of the log object but they need to include the
    following:
     - A method called when updating that contains the *args argument.
     - A method for copying the log object.

    Methods
    -------
    update(*args)
        Method called when updating the log object.
    copy()
        Method for copying the log object.

    Attributes
    ----------
    data
        The data collected by the log object.
    epoch
        The epoch of the data collected by the log object.

    Attributes can be added but need to be updated locally using a global variable or can be updated
    in the genetic algorithm when a child instance of genetic_algortihm is created with a custom run() method.
    """
    def __init__(self, b2num, bitsize, b2nkwargs, *args, **kwargs):
        self.epoch = []
        self.data = []

        self.bitsize = bitsize
        self.b2n = b2num
        self.b2nkwargs = b2nkwargs

        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, item: int):
        return self.data[item]

    def __copy__(self):
        object_copy = log_object(self.b2n, self.bitsize, *self.args, **self.kwargs)
        object_copy.data = self.data
        object_copy.epoch = self.epoch
        return object_copy

    def __len__(self):
        return len(self.epoch)

    def copy(self):
        return self.__copy__()

    def save2txt(self, data: str, filename: str, sep: str = ";"):
        """
        Save the log object to a text file.

        :type data: str
        :param data: The data to save in the form of the log_object attr.

        :type filename: str
        :param filename: The filename of the text file.

        :type sep: str
        :param sep: The separator used in the text file.

        :return: None
        """

        # get the data
        try:
            data = np.array(getattr(self, data)).T
        except AttributeError:
            raise AttributeError("The data '%s' does not exist in the log object '%s'." % (data, self.__class__.__name__))

        # Save to text file with separator sep
        with open(filename, "w") as f:
            f.write(sep.join(str(i) for i in range(len(self.epoch))) + "\n")
            for i in range(len(data)):
                f.write(sep.join(str(data[i, j]) for j in range(data.shape[1])) + "\n")

        return None


    def update(self, data, *args):
        self.data.append(data)
        self.epoch.append(len(self.data))
        return None



class log_time(log_object):
    """
    A log object for logging the time of the GA.

    Methods
    -------
    update(*args)
        Method called when updating the log object.
    copy()
        Method for copying the log object.

    Attributes
    ----------
    data
        The time data collected by the log object.
    epoch
        The epoch of the data collected by the log object.

    """

    def __init__(self, *args, **kwargs):
        self.calculation = []
        super().__init__(*args, **kwargs)

    def __copy__(self):
        object_copy = log_time(self.b2n, self.bitsize, *self.args, **self.kwargs)
        object_copy.data = self.data
        object_copy.epoch = self.epoch
        object_copy.calculation = self.calculation
        return object_copy

    def update(self, data, *args):
        """
        Update the log object.

        :param data: The time data to be added to the log object.
        :param args: Calculation number.
        :return: None
        """
        self.data.append(data)

        # Get the epoch number from the length of the data list.
        self.epoch.append(len(self.data))

        # Get the calculation number from the args and subtract from previous amount to get the number for this epoch.
        # if len(self.epoch) > 1:
        #     self.calculation.append(args[0] - self.calculation[-1])
        # else:
        #

        self.calculation.append(args[0])
        return None

    def plot(self, save_as="", show=True, *args, **kwargs):
        """
        Plot the time data.
        :param save_as: The path to save the plot to.
        :param show: Show the plot.
        :param args: Arguments for the plot.
        :param kwargs: Keyword arguments for the plot.

        :return: The plot in a AdrianPackv402.Default object.
        See documentation (AdrianPack on github) for more information.
        """
        linestyle = "-"
        if "linestyle" in kwargs:
            linestyle = kwargs["linestyle"]
            kwargs["linestyle"].pop()

        marker = "o"
        if "marker" in kwargs:
            marker = kwargs["marker"]
            kwargs["marker"].pop()

        kwargs["x_label"] = "epoch"
        kwargs["y_label"] = "time"

        kwargs["data_label"] = "avg per epoch: %s s" % (sum(np.array(self.data)[1:] - np.array(self.data)[:-1]) / len(self.data))

        pl = Default(self.epoch, self.data, *args, linestyle=linestyle, marker=marker, **kwargs)

        plt.title("Time: %s s" % self.data[-1])

        if show:
            if save_as != "":
                pl.save_as = save_as
            pl()

        return pl



class log_selection(log_object):
    """
    A log object for logging the selection of the GA.

    Methods
    -------
    update(*args)
        Method called when updating the log object.
    copy()
        Method for copying the log object.

    Attributes
    ----------
    data
        The selection data collected by the log object.
    epoch
        The epoch of the data collected by the log object.
    probability
        The probability of the selection data collected by the log object.
    fitness
        The fitness of the selection data collected by the log object.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probabilty = []
        self.fitness = []

    def __copy__(self):
        object_copy = log_selection(self.b2n, self.bitsize, self.b2nkwargs, *self.args, **self.kwargs)
        object_copy.data = self.data
        object_copy.epoch = self.epoch
        object_copy.probabilty = self.probabilty
        object_copy.fitness = self.fitness
        return object_copy

    def update(self, data, *args):
        self.data.append(data)
        self.epoch.append(len(self.data))
        self.probabilty.append(args[0])
        self.fitness.append(args[1])
        return None

    def __getitem__(self, item):
        return self.fitness[item]

    def plot(self, top: int = None, show: bool = True, save_as: str = "",**kwargs):
        if top == None:
            top = len(self.data[0])

        avgpepoch = [np.average(i[:top]) for i in self.fitness]

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

class log_ranking(log_object):
    """
    A log object for logging the ranking of the GA.

    Methods
    -------
    update(*args)
        Method called when updating the log object.
    copy()
        Method for copying the log object.

    Attributes
    ----------
    data
        The ranking data collected by the log object.
    epoch
        The epoch of the data collected by the log object.
    ranknum
        The numerical variables [x1, x2, ...., xn] in order of rank.
    distance
        The distance between the numerical variables and the target.
    distancefx
        The distance between the result of numerical variables and the target.
    effectivity
        The effectivity of the ranking data collected by the log object.
        calculated as: by taking the distance between the optimium or minimum and the result of the solutions in ranknum.
    bestsol
        The best solution found by the GA per epoch

    """
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranknum = []
        self.result = []
        self.bestresult = []
        self.effectivity = []
        self.distancex = []
        self.distancefx = []
        self.bestsol = []

    def __copy__(self):
        object_copy = log_ranking(self.b2n, self.bitsize, *self.args, **self.kwargs)
        object_copy.data = self.data
        object_copy.epoch = self.epoch
        object_copy.ranknum = self.ranknum
        object_copy.effectivity = self.effectivity
        object_copy.distancex = self.distancex
        object_copy.bestsol = self.bestsol
        object_copy.distancefx = self.distancefx
        object_copy.bestresult = self.bestresult
        object_copy.result = self.result
        return object_copy

    def update(self, data, *args):
        """

        :param data:
        :param args: fx, the optimum, highest effectivity[x coordinates, fx]

        :return:
        """
        self.data.append(data)
        self.epoch.append(len(self.data))
        self.fx = args[0]

        self.ranknum.append(self.b2n(data, self.bitsize, **self.b2nkwargs))

        self.result.append(self.fx)

        # Calculate the euclidian distance between the optimum and all pop values
        self.distancefx.append(self.fx - args[2])

        # Determine the effectivity relative to the largest distcance after the first epoch.
        self.effectivity.append(1 - self.distancefx[-1] / max(self.distancefx[-1]))

        # Find the best solution of this iteration and append it to the list
        ind = np.where(self.effectivity[-1] == max(self.effectivity[-1]))[0]
        self.bestsol.append(self.ranknum[-1][ind])
        self.bestresult.append(self.fx[ind])

    def __getitem__(self, item: int):
        return self.ranknum[item]

    def __repr__(self):
        return str(self.ranknum)

    def plot(self, top: int = None, show: bool = True, save_as: str = ""
             , datasource = None, **kwargs):

        if top == None:
            top = len(self.effectivity[0])

        if datasource == None:
            datasource = self.distancefx

        elif datasource == self.bestsol:
            pass

        avgpepoch = [np.average(i[0:]) for i in datasource]

        if not "linestyle" in kwargs:
            kwargs["linestyle"] = "-"

        if not "marker" in kwargs:
            kwargs["marker"] = "o"

        pl = Default(self.epoch, avgpepoch, **kwargs)
        if show:
            if save_as != "":
                pl.save_as = save_as
            pl()
        return pl


class log_value(log_object):
    """
    A log object for logging the value of the GA.

    Methods
    -------
    update(*args)
        Method called when updating the log object.
    copy()
        Method for copying the log object.

    Attributes
    ----------
    data
        The value data collected by the log object.
    epoch
        The epoch of the data collected by the log object.
    value
        The binary value of the solutions proposed by the ga per epoch
    numvalue
        The numerical value of the solutions proposed by the ga per epoch
    topx
        The top x solutions proposed by the ga per epoch

    """
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
                                                     bitsize=self.bitsize,
                                                     **self.b2nkwargs))[:, 0])
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

        plt.plot(self.numvalue[epoch][:, 0], self.numvalue[epoch][:, 1],
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

        linsp = np.linspace(low, high, 10000)
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

    def animate2d(self, tfx: Callable, low: int, high: int, save_as="", epochs=0,
                  fitness=None):
        """

        :param tfx:
        :param low:
        :param high:
        :param save_as:
        :return:
        """

        figure, axis1 = plt.subplots(1, 1)

        print(fitness)
        line2data = np.array(fitness)
        if fitness==None:
            line2data = np.asarray(self.numvalue)


        line, = axis1.plot(self.numvalue[0][:, 0], self.numvalue[0][:, 1],
                         linestyle="",
                         marker="o", label="Algortithm")

        # line2 = axis2.bar(range(len(np.sort(line2data[0, :]))), np.sort(line2data[0, :]))

        x1, x2 = np.linspace(low, high, 1000), np.linspace(low, high, 1000)
        X1, X2 = np.meshgrid(x1, x2)
        y = tfx([X1, X2])

        pcmesh = axis1.pcolormesh(X1, X2, y, cmap='RdBu', shading="auto")

        axis1.set_xlim(low, high)
        axis1.set_ylim(low, high)

        axis1.legend(loc="upper right")

        divider = make_axes_locatable(axis1)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        figure.colorbar(pcmesh, cax=cax, orientation='vertical')

        def update(frame):
            print(frame)
            line.set_data(self.numvalue[frame][:, 0], self.numvalue[frame][:, 1])

            # sorted_fitness = np.sort(line2data[frame, :])

            # for i, b in enumerate(line2):
            #     b.set_height(sorted_fitness[i])


            axis1.set_title("2D plot")
            # axis2.set_title("fitness of individuals")
            plt.title("Iteration: %s" % frame)
            return None

        if epochs == 0:
            epochs = range(len(self.epoch))

        animation = FuncAnimation(figure, update, interval=500,
                                  frames=epochs)

        if save_as == "":
            plt.show()
            return None

        animation.save(save_as, dpi=600, writer=PillowWriter(fps=1))

        return None

if __name__ == "__main__":
    pass


