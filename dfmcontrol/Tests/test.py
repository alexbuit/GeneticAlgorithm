
from typing import Callable, Iterable
from time import time
import numpy as np
import pytest
import unittest

from dfmcontrol.Helper import *
from dfmcontrol.Utility import crossover, pop, selection

# @pytest.mark.usefixtures("db_class")
class TestDFM(unittest.TestCase):

    def test_convert_bin64(self):

        randarr = np.linspace(0, 100, 10000, dtype=np.double)
        binarr64 = float2NdbitIEEE754(randarr, 64)
        floatarr64 = Ndbit2floatIEEE754(binarr64, 64)

        for i in range(randarr.size):
            self.assertEqual(randarr[i], floatarr64[i])

    def test_convert_bin32(self):

        randarr = np.linspace(0, 100, 10000, dtype=float)
        binarr32 = float2NdbitIEEE754(randarr, 32)
        floatarr32 = Ndbit2floatIEEE754(binarr32, 32)

        for i in range(randarr.size):
            self.assertAlmostEqual(randarr[i], floatarr32[i], places=5)

    def test_crossover_single_point(self):

        def fitness(x):
            return x

        test_pop = pop.uniform_bit_pop([1000, 16], 16, [0, 100])
        parents = selection.roulette_selection(test_pop, fitness, 2, ndbit2int, bitsize=16)
        crossed = crossover.IEEE_single_point(parents, 0.5, bitsize=16)

        self.assertEqual(crossed.shape, (2, 16))

    def test_pop_gen(self):

        from dfmcontrol.Utility.pop import _create_pop
        from numpy.random import normal
        from scipy.stats import cauchy

        pop = _create_pop(shape=(10, 10), pop_float=cauchy.rvs, pop_kwargs={"loc": 0, "scale": 1, "size": 100},
                       n2b=int2ndbit, n2bkwargs={"factor": 1, "bitsize": 10})

        assert pop.shape == (10, 100) # 10 Individuals, 10 genes with 10 bits per gene

