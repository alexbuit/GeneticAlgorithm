import numpy as np

from dfmcontrol.AdrianPackv402 import Fileread
from dfmcontrol.AdrianPackv402 import Aplot

data = Fileread.Fileread(r"results.txt", dtype=float)()
data = list(data.values())

datamat = np.array(data)

# calculate the average fitness
avgfit = np.average(datamat, axis=1)
# calculate the standard deviation
stdfit = np.std(datamat, axis=1)

# calculate the minimum fitness
minfit = np.min(datamat, axis=1)
# calculate the maximum fitness
maxfit = np.max(datamat, axis=1)

# plot the average fitness
plmin = Aplot.Default(np.arange(len(minfit)), minfit, colour="C1", data_label="Minimum fitness", legend_loc="upper right")

pl = Aplot.Default(np.arange(len(avgfit)), avgfit, colour="C0", data_label="Average fitness", add_mode=True)
plmax = Aplot.Default(np.arange(len(maxfit)), maxfit, colour="C2", data_label="Maximum fitness", add_mode=True)

plmin += pl
plmin += plmax

plmin()

