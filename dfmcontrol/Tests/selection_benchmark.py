import matplotlib.pyplot as plt
import numpy as np

import time as t

from dfmcontrol.Helper import *
from dfmcontrol.Utility import crossover, pop, selection

def select(y, method):
    def fitness(x):
        return sum(x)

    test_pop = pop.uniform_bit_pop([16, y], 16, [0, 1])
    parents, _, _, _ = selection.rank_selection(test_pop, fitness, 16, ndbit2int, b2nkwargs={"factor": 1})

