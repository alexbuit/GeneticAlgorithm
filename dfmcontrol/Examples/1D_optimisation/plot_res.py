import numpy as np

from dfmcontrol.AdrianPackv402 import Fileread
from dfmcontrol.AdrianPackv402 import Aplot

data = Fileread.Fileread(r"results.txt", dtype=float)()
data = list(data.values())
print(data)

# plot the final epoch
pl = Aplot.Default(np.arange(len(data[-1])), data[-1], add_mode=True, colour="C1", data_label="Final epoch")
pl1 = Aplot.Default(np.arange(len(data[0])), data[0], x_label="Individual", y_label="fitness", data_label="First epoch", colour="C0")
pl1 += pl
pl1()