from typing import Callable, Iterable
from time import time
import numpy as np
import pytest
import unittest

from dfmcontrol.Helper import *
import dfmcontrol.Utility.pop as pop

import inspect

def get_methods_count(file_path):
    with open(file_path) as file:
        lines = file.readlines()
        methods_count = 0
        for line in lines:
            if 'def ' in line:
                methods_count += 1
        return methods_count

# @pytest.mark.usefixtures("db_class")
class Test_create_pop(unittest.TestCase):

    bitsize = 32
    loc = 0
    scale = 1
    factor = 1
    bias = 0

    low = 0
    high = 1

    def test_amount_of_tests_in_suite(self):
        """
        Verify if the amount of tests is equal to the ones prescribed in the
        documenation.

        For _create_pop this amount is 7

        Tests included:
         - One individual, ten genes [1, 10]
         - Ten individuals, one gene [10, 1]
         - Ten individuals, ten genes [10, 10]
         - One thousand individuals, one thousand genes [1000, 1000]

         - cauchy distribution; shape [10, 10]
         - uniform distribution; shape [10, 10]
         - normal distribution; shape [10, 10]

        Standard parameters
         - bitsize = 32
         - loc = 0
         - scale = 1
         - factor = 1
         - bias = 0
         - low = 0
         - high = 1
        """
        methods = [m for m in Test_create_pop.__dict__ if
                   callable(getattr(Test_create_pop, m))]

        methods.remove("test_amount_of_tests_in_suite")
        methods.remove("equality_check")
        methods.remove("_UnitTestCase__pytest_class_setup")

        assert len(methods) == 7

    def test_normal(self):
        from numpy.random import normal

        for _ in range(10):
            shape = (10, 10)
            size = shape[0] * shape[1]
            bitsize = self.bitsize

            popul = pop._create_pop(shape=shape, pop_float=normal, pop_kwargs={"loc": self.loc, "scale": self.scale, "size": size},
                              n2b=int2ndbit, n2bkwargs={"factor": self.factor,"bias": self.bias ,"bitsize": bitsize})
            popul_IEEE = pop._create_pop(shape=shape, pop_float=normal, pop_kwargs={"loc": self.loc, "scale": self.scale, "size": size},
                                n2b=pop.float2NdbitIEEE754, n2bkwargs={"bitsize": bitsize})

            self.equality_check([popul, popul_IEEE],
                                (shape[0], shape[1] * bitsize))

    def test_cauchy(self):
        from scipy.stats import cauchy

        shape = (10, 10)
        size = shape[0] * shape[1]
        bitsize = self.bitsize

        for _ in range(10):
            popul = pop._create_pop(shape=shape, pop_float=cauchy.rvs, pop_kwargs={"loc": self.loc, "scale": self.scale, "size": size},
                              n2b=int2ndbit, n2bkwargs={"factor": self.factor, "bitsize": bitsize})
            popul_IEEE = pop._create_pop(shape=shape, pop_float=cauchy.rvs, pop_kwargs={"loc": self.loc, "scale": self.scale, "size": size},
                                n2b=pop.float2NdbitIEEE754, n2bkwargs={"bitsize": bitsize})

            self.equality_check([popul, popul_IEEE],
                                (shape[0], shape[1] * bitsize))

    def test_uniform(self):
        from numpy.random import uniform

        shape = (10, 10)
        size = shape[0] * shape[1]
        bitsize = self.bitsize

        for _ in range(10):
            popul = pop._create_pop(shape=shape, pop_float=uniform, pop_kwargs={"low": self.low, "high": self.high, "size": size},
                              n2b=int2ndbit, n2bkwargs={"factor": self.factor, "bitsize": bitsize})
            popul_IEEE = pop._create_pop(shape=shape, pop_float=uniform, pop_kwargs={"low": self.low, "high": self.high, "size": size},
                                n2b=pop.float2NdbitIEEE754, n2bkwargs={"bitsize": bitsize})

            self.equality_check([popul, popul_IEEE],
                                (shape[0], shape[1] * bitsize))

    def test_one_ten(self):

        shape = (1, 10)
        size = shape[0] * shape[1]
        bitsize = self.bitsize

        for _ in range(10):

            popul = pop._create_pop(shape=shape, pop_float=np.random.uniform, pop_kwargs={"low": self.low, "high": self.high, "size": size},
                              n2b=int2ndbit, n2bkwargs={"factor": self.factor, "bitsize": bitsize})
            popul_IEEE = pop._create_pop(shape=shape, pop_float=np.random.uniform, pop_kwargs={"low": self.low, "high": self.high, "size": size},
                                n2b=pop.float2NdbitIEEE754, n2bkwargs={"bitsize": bitsize})

            self.equality_check([popul, popul_IEEE],
                                (shape[0], shape[1] * bitsize))

    def test_ten_one(self):

        shape = (10, 1)
        size = shape[0] * shape[1]
        bitsize = self.bitsize

        for _ in range(10):
            popul = pop._create_pop(shape=shape, pop_float=np.random.uniform, pop_kwargs={"low": self.low, "high": self.high, "size": size},
                              n2b=int2ndbit, n2bkwargs={"factor": self.factor, "bitsize": bitsize})
            popul_IEEE = pop._create_pop(shape=shape, pop_float=np.random.uniform, pop_kwargs={"low": self.low, "high": self.high, "size": size},
                                n2b=pop.float2NdbitIEEE754, n2bkwargs={"bitsize": bitsize})

            self.equality_check([popul, popul_IEEE],
                                (shape[0], shape[1] * bitsize))

    def test_ten_ten(self):

        shape = (10, 10)
        size = shape[0] * shape[1]
        bitsize = self.bitsize

        for _ in range(10):
            popul = pop._create_pop(shape=shape, pop_float=np.random.uniform, pop_kwargs={"low": self.low, "high": self.high, "size": size},
                              n2b=int2ndbit, n2bkwargs={"factor": self.factor, "bitsize": bitsize})
            popul_IEEE = pop._create_pop(shape=shape, pop_float=np.random.uniform, pop_kwargs={"low": self.low, "high": self.high, "size": size},
                                n2b=pop.float2NdbitIEEE754, n2bkwargs={"bitsize": bitsize})

            self.equality_check([popul, popul_IEEE],
                                (shape[0], shape[1] * bitsize))

    def test_thousand_thousand(self):

        shape = (1000, 1000)
        size = shape[0] * shape[1]
        bitsize = self.bitsize

        popul = pop._create_pop(shape=shape, pop_float=np.random.uniform, pop_kwargs={"low": self.low, "high": self.high, "size": size},
                          n2b=int2ndbit, n2bkwargs={"factor": self.factor, "bitsize": bitsize})
        popul_IEEE = pop._create_pop(shape=shape, pop_float=np.random.uniform, pop_kwargs={"low": self.low, "high": self.high, "size": size},
                            n2b=pop.float2NdbitIEEE754, n2bkwargs={"bitsize": bitsize})

        self.equality_check([popul, popul_IEEE], (shape[0], shape[1]*bitsize))

    def equality_check(self, populations, expectation):
        for i in populations:
            assert i.shape == expectation
            assert i.dtype == np.uint8
            assert i.ndim == 2
            assert isinstance(i, np.ndarray)


class Test_rand_bit_pop(unittest.TestCase):
    bitsize = 32
    loc = 0
    scale = 1
    factor = 1
    bias = 0

    low = 0
    high = 1

    def test_amount_of_tests(self):
        """
        Test if the amount of tests is correct

        The tests for this class are:
         - One individual, ten genes [1, 10]
         - Ten individuals, one gene [10, 1]
         - Ten individuals, ten genes [10, 10]
         - One thousand individuals, one thousand genes [1000, 1000]

        """
        methods = [m for m in Test_rand_bit_pop.__dict__ if
                   callable(getattr(Test_rand_bit_pop, m))]

        methods.remove("test_amount_of_tests")
        methods.remove("equality_check")
        methods.remove("_UnitTestCase__pytest_class_setup")

        assert len(methods) == 4
    def test_one_ten(self):

        shape = (1, 10)
        bitsize = self.bitsize

        for _ in range(10):
            popul = pop.rand_bit_pop(n=shape[0], m=shape[1]*bitsize)

            self.equality_check(popul, (shape[0], shape[1]*bitsize))

    def test_ten_one(self):

        shape = (10, 1)
        bitsize = self.bitsize

        for _ in range(10):
            popul = pop.rand_bit_pop(n=shape[0], m=shape[1]*bitsize)

            self.equality_check(popul, (shape[0], shape[1]*bitsize))

    def test_ten_ten(self):

        shape = (10, 10)
        bitsize = self.bitsize

        for _ in range(10):
            popul = pop.rand_bit_pop(n=shape[0], m=shape[1]*bitsize)

            self.equality_check(popul, (shape[0], shape[1]*bitsize))

    def test_thousand_thousand(self):

        shape = (1000, 1000)
        bitsize = self.bitsize

        popul = pop.rand_bit_pop(n=shape[0], m=shape[1]*bitsize)

        self.equality_check(popul, (shape[0], shape[1]*bitsize))

    def equality_check(self, populations, expectation):
        assert populations.shape == expectation
        assert populations.dtype == np.uint8
        assert populations.ndim == 2
        assert isinstance(populations, np.ndarray)

class Test_normal_IEEE(unittest.TestCase):

    bitsize = 32
    loc = 0
    scale = 1
    factor = 1
    bias = 0

    low = 0
    high = 1

    def test_amount_of_tests(self):
        """
        Test if the amount of tests is correct

        The tests for this class are:
         - One individual, ten genes [1, 10]
         - Ten individuals, one gene [10, 1]
         - Ten individuals, ten genes [10, 10]
         - One thousand individuals, one thousand genes [1000, 1000]

        """
        methods = [m for m in Test_rand_bit_pop.__dict__ if
                   callable(getattr(Test_rand_bit_pop, m))]

        methods.remove("test_amount_of_tests")
        methods.remove("equality_check")
        methods.remove("_UnitTestCase__pytest_class_setup")

        assert len(methods) == 4


    def equality_check(self, populations, expectation):
        assert populations.shape == expectation
        assert populations.dtype == np.uint8
        assert populations.ndim == 2
        assert isinstance(populations, np.ndarray)

    def test_one_ten(self):

        shape = (1, 10)
        bitsize = self.bitsize

        for _ in range(10):
            popul = pop.normalrand_bit_pop_IEEE(shape[0], bitsize, self.low, self.high)

            self.equality_check(popul, (shape[0], shape[1]*bitsize))

    def test_ten_one(self):

        shape = (10, 1)
        bitsize = self.bitsize

        for _ in range(10):
            popul = pop.normalrand_bit_pop_IEEE(shape[0], bitsize, self.low, self.high)

            self.equality_check(popul, (shape[0], shape[1]*bitsize))

    def test_ten_ten(self):

        shape = (10, 10)
        bitsize = self.bitsize

        for _ in range(10):
            popul = pop.normalrand_bit_pop_IEEE(shape[0], bitsize, self.low, self.high)

            self.equality_check(popul, (shape[0], shape[1]*bitsize))


    def test_thousand_thousand(self):

        shape = (1000, 1000)
        bitsize = self.bitsize

        popul = pop.normalrand_bit_pop_IEEE(shape[0], bitsize, self.low, self.high)

        self.equality_check(popul, (shape[0], shape[1]*bitsize))

