import numpy as np

from dfmcontrol.Helper import *
from dfmcontrol.Utility import crossover, pop, selection

def cross():
    def fitness(x):
        return sum(x)

    test_pop = pop.uniform_bit_pop([4, 4], 16, [0, 1])
    parents, _, _, _ = selection.rank_selection(test_pop, fitness, 16, ndbit2int, b2nkwargs={"factor": 1})

    children = []
    for p in parents:
        c1, c2 = crossover.IEEE_equal_prob_cross(test_pop[p[0]], test_pop[p[1]], bitsize=16)
        children.append(c1)
        children.append(c2)
    print(fitness(np.asarray(convertpop2n(ndbit2int, children, 16))) - fitness(np.asarray(convertpop2n(ndbit2int, test_pop, 16))))

    return np.asarray(convertpop2n(ndbit2int, children, 16)) - np.asarray(convertpop2n(ndbit2int, test_pop, 16))

for i in range(6):
    cross()
