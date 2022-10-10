
from typing import Callable

import numpy as np
import math
import random as rand

# Population size
n = 1000
# Bit size
m = 16

randbitpop = [np.random.randint(0, 2, size=m) for _ in range(n)]

def GA(fx: Callable, pop: np.ndarray, max_iter: int, select: Callable,
       cross: Callable, mutate:Callable):

    for _ in range(max_iter):
        # Parents function should return pairs of parent indexes in pop
        parents = select(pop, fx)
        # Apply the cross function to all parent couples
        children = [cross(pop[p[0]], pop[p[1]]) for p in parents]
        # Mutate the population (without elitism)
        pop = mutate(children)

    return pop

