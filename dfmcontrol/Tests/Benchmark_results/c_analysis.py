
from dfmcontrol.AdrianPackv402 import Fileread
from dfmcontrol.AdrianPackv402 import Aplot

data = Fileread.Fileread(r"crossover_benchmarksC.txt", dtype=float, delimiter=" ")()
datapython = Fileread.Fileread(r"crossover_benchmarksPYTHON.txt", dtype=float, delimiter=" ", head=False)()

data = list(data.values())
datapython = list(datapython.values())

data[1] = [x / 100000 * 1000 for x in data[1]]
data[2] = [x / 100000 * 1000 for x in data[2]]
data[3] = [x / 100000 * 1000 for x in data[3]] # to per iteration times milliseconds

pl_single = Aplot.Default(data[0], data[1], colour="C0", data_label="Single point crossover in C", legend_loc="upper left", x_label="Number of genes in an individual", y_label="Time per iteration (ms)", marker="x")
pl_double = Aplot.Default(data[0], data[2], colour="C1", data_label="Double point crossover in C", add_mode=True, marker="x")
pl_uniform = Aplot.Default(data[0], data[3], colour="C2", data_label="Uniform crossover in C", add_mode=True, marker="x")

pl_singlePython = Aplot.Default(datapython[0], datapython[1], colour="C0", data_label="Single point crossover in Python", add_mode=True, linestyle="--", marker="p")
pl_doublePython = Aplot.Default(datapython[0], datapython[2], colour="C1", data_label="Double point crossover in Python", add_mode=True, linestyle="--", marker="p")
pl_uniformPython = Aplot.Default(datapython[0], datapython[3], colour="C2", data_label="Uniform crossover in Python", add_mode=True, linestyle="--", marker="p")

# pl_single += pl_singlePython
# pl_single += pl_doublePython
# pl_single += pl_uniformPython

# pl_single += pl_double
pl_single += pl_uniform

pl_single()