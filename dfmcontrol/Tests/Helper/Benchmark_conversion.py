
import dfmcontrol as dfm

import numpy as np
import matplotlib.pyplot as plt

import time as t
# Initialise the population

time_int2bit = []
time_bit2int = []

for i in range(1, 40):

    pop = dfm.Utility.pop.uniform_bit_pop([16, i], 16, [0, 1])

    # Convert the population to integers
    tstart = t.time()
    for _ in range(1000):
        pop_int = dfm.Helper.ndbit2int(pop, 16, factor=5, bias=0, normalised=True)
    time_bit2int.append(t.time() - tstart)

    # Convert the population to bits
    tstart = t.time()
    for _ in range(1000):
        pop_bit = dfm.Helper.int2ndbit(pop_int, 16, factor=5, bias=0, normalised=True)
    time_int2bit.append(t.time() - tstart)

    print(i, time_bit2int[-1], time_int2bit[-1])

# write the results to a file
arr = np.asarray([time_bit2int, time_int2bit])
np.savetxt("benchmark_conversion.txt", arr, delimiter=";")
