import numpy as np

from dfmcontrol.Helper import ndbit2int, int2ndbit

individuals = 40
genes = 50

factor = 15
bias = 5
normalised = True

tested_bitsizes = np.arange(4, 48)

# initiate a random population
pop = np.random.randint(-5, 5, size=(individuals, genes)).astype(float)
popcopy = pop.copy()

notequal = []

for bitsize in tested_bitsizes:
    # convert to ndbit
    popint = int2ndbit(popcopy, bitsize=bitsize, factor=factor, bias=bias, normalised=normalised)

    # convert back to int
    pop = ndbit2int(popint, bitsize=bitsize, factor=factor, bias=bias, normalised=normalised)

    # get the amount of genes that are not equal

    amount_of_errors = np.sum(np.around(pop) != np.around(popcopy))
    error_rate = amount_of_errors / (individuals * genes)

    notequal.append(error_rate)

# Save the data to a file
np.savetxt("error_rate_zoom_PYTHON.txt", np.array([tested_bitsizes, notequal]).T, delimiter=",", header="bitsize, error_rate")

