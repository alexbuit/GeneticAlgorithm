import unittest

from typing import Callable, Iterable
from time import time
import numpy as np

from helper import *


class TestDFM(unittest.TestCase):

    def test_convert_bin64(self):

        randarr = np.linspace(0, 100, 10000, dtype=np.double)
        binarr64 = float2Ndbit(randarr, 64)
        floatarr64 = Ndbit2float(binarr64, 64)

        for i in range(randarr.size):
            self.assertEqual(randarr[i], floatarr64[i])

    def test_convert_bin32(self):

        randarr = np.linspace(0, 100, 10000, dtype=float)
        binarr32 = float2Ndbit(randarr, 32)
        floatarr32 = Ndbit2float(binarr32, 32)

        for i in range(randarr.size):
            self.assertAlmostEqual(randarr[i], floatarr32[i], places=5)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestDFM("test_convert_bin64"))
    suite.addTest(TestDFM("test_convert_bin32"))
    return suite


def time_t(fx: Callable, params: Iterable):
    tlist = [time(),]
    for p in params:
        fx(p)
        tlist.append(time() - tlist[-1])

    return tlist

if __name__ == '__main__':
    # runner = unittest.TextTestRunner()
    # runner.run(suite())

    print(time_t(float2Ndbit, [(np.array([0.213, 0.213, 0.4234]), 64)]))