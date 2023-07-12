
from dfmcontrol.Helper import *
from dfmcontrol.Utility import crossover, pop, selection

def cross():
    def fitness(x):
        return x

    test_pop = pop.uniform_bit_pop([10, 4], 16, [0, 1])
    parents, _, _, _ = selection.rank_selection(test_pop, fitness, 16, ndbit2int, b2nkwargs={"factor": 1})

    print(parents)

    for p in parents:
        print(p)
        crossed = crossover.single_point(test_pop[p[0]], test_pop[p[1]], bitsize=16)
    return len(crossed)

print(cross())