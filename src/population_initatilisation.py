
import numpy as np

from typing import Union, Iterable
from scipy.stats import cauchy
from helper import *

def rand_bit_pop(n: int, m: int) -> np.ndarray:
    """
    Generate a random bit population
    :param n: Population size dtype int
    :param m: Bitsize dtype int
    :return: List of random bits with a bit being a ndarray array of 0 and 1.
    """
    return np.array([np.random.randint(0, 2, size=m) for _ in range(n)])


def normalrand_bit_pop_float(n, bitsize, lower, upper):

    pop_float = np.linspace(lower, upper, num=n)
    blist = []
    if bitsize == 32:
        for val in range(pop_float.size):
            blist.append(floatToBinary32(pop_float[val]))
            # tval, tres = pop_float[val], b2sfloat(floatToBinary64(pop_float[val]))[0]
            # try: np.testing.assert_almost_equal(tres, tval )
            # except AssertionError: print("Fail")

    elif bitsize == 64:
        for val in range(pop_float.size):
            blist.append(floatToBinary64(pop_float[val]))
            # tval, tres = pop_float[val], b2dfloat(blist[-1])[0]
            # try: np.testing.assert_almost_equal(tres, tval )
            # except AssertionError: print("Fail")

    else:
        pass
    return np.array(blist)


def cauchyrand_bit_pop_float(shape: Union[Iterable, float], bitsize: int, loc: float,
                             scale: float) -> np.ndarray:
    if isinstance(shape, int):
        shape = (shape, 1)
    elif len(shape) == 1:
        shape = (shape[0], 1)

    size = shape[0] * shape[1]

    pop_float = cauchy.rvs(loc=loc, scale=scale, size=size)
    pop_float = np.array(np.array_split(pop_float, int(size/shape[0])), dtype=float)
    blist = []
    for val in range(pop_float.shape[1]):
        blist.append(float2Ndbit(pop_float[:, val], bitsize))

    return np.array(blist)


def uniform_bit_pop_float(shape: Union[Iterable, float], bitsize: int, low: float,
                             high: float) -> np.ndarray:

    if isinstance(shape, int):
        shape = (shape, 1)
    elif len(shape) == 1:
        shape = (shape[0], 1)

    size = shape[0] * shape[1]

    pop_float = np.random.uniform(low, high, size)
    pop_float = np.array(np.array_split(pop_float, int(size/shape[0])), dtype=float)
    blist = []
    for val in range(pop_float.shape[1]):
        blist.append(float2Ndbit(pop_float[:, val], bitsize))

    return np.array(blist)


def bit8(shape: list):

    if isinstance(shape, int):
        shape = (shape, 1)
    elif len(shape) == 1:
        shape = (shape[0], 1)

    shape[1] *= 8

    blist = []
    for val in range(shape[0]):
        blist.append(np.random.randint(0, 2, shape[1]))

    return np.array(blist, dtype=np.uint8)

# print(bit8([10, 2]))