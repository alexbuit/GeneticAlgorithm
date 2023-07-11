
from dfmcontrol.AdrianPackv402.Aplot import Default
import numpy as np

# Michelawizc minima
data = np.array([-1.8013034, -2.7603947, -3.6988571, -4.6876582, -5.6876582, -6.6808853, -7.6637574, -8.6601517, -9.6601517])
data_extrapolated = np.array([i*-0.99864 + 0.30271 for i in range(1, 11)])

pl1 = Default(list(range(2, 11)), data, x_label='n-dimension', y_label='optimmum', data_label='Michelawizc minima', legend_loc='upper right')
pl2 = Default(list(range(1, 11)), data_extrapolated, x_label='x', y_label='y', data_label='Michelawizc minima extrapolated',
              add_mode=True, degree=1, colour="red")
pl1 += pl2
fit = pl2.lambdify_fit()
print(pl2.fit_stats())
pl1()

dif2 = np.abs([fit(i) - data[i-2] for i in range(2, 11)])
dif = np.abs(data - data_extrapolated[1:])

pl3 = Default(list(range(2, 11)), dif, x_label='dimension', y_label='|exact - predicted|', linestyle="",
              data_label="difference")

pl4 = Default(list(range(2, 11)), dif2, x_label='dimension', y_label='|exact - predicted|', linestyle="",
              data_label="difference", add_mode=True, colour="r")

pl3 += pl4
pl3()