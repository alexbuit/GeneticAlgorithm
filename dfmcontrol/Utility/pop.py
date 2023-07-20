
import numpy as np

from typing import Union, Iterable, List
from scipy.stats import cauchy

try:
    from dfmcontrol.Helper import *
except ImportError:
    from dfmcontrol.Helper import *


def _create_pop(**kwargs):
    """
    Generate a population of individuals.

    :param kwargs: Keyword arguments for the population generation.
    :param kwargs["shape"]: Shape of the population dtype tuple.
    :param kwargs["pop_float"]: Method for creating the randomized population array
    :param kwargs["pop_kwargs"]: Keyword arguments for the creation method.
    :param kwargs["n2b"]: Method for converting the numerical values (or genes) to a binary array (individual).

    :return: List of individuals.
    """

    shape = kwargs.get("shape", None)
    individuals, variables = shape
    size = shape[0] * shape[1]

    pop_float = kwargs.get("pop_float", None)(**kwargs.get("pop_kwargs", None))

    factor = kwargs.get("n2bkwargs", None).get("factor", np.abs(np.max(pop_float)))
    bias = kwargs.get("n2bkwargs", None).get("bias", 0)

    pop_float = pop_float / np.abs(np.max(pop_float)) * factor + bias

    pop_float = np.array(np.array_split(pop_float, int(individuals)), dtype=float)

    barr = kwargs.get("n2b", None)(pop_float, **kwargs.get("n2bkwargs", None)).astype(np.uint8)

    if barr.ndim == 1:
        barr = np.array([barr], dtype=np.uint8)

    return barr

def rand_bit_pop(n: int, m: int) -> np.ndarray:
    """
    Generate a random bit population, n individuals with m variables.

    :param n: Population size dtype int
    :param m: Bitsize dtype int

    :return: List of random bits with a bit being a ndarray array of 0 and 1.
    """
    return np.array([np.random.randint(0, 2, size=m) for _ in range(n)])


# float2NdbitIEEE754 and NdbittofloatIEEE754 routines

def normalrand_bit_pop_IEEE(n, bitsize, lower, upper):  # TODO: implement multi variable
    """
    Generate a normal distributed bit population with floats converted with
    float2NdbitIEEE754 and NdbittofloatIEEE754.

    :param n: Population size dtype int individuals
    :param bitsize: Bitsize dtype int
    :param lower: Lower bound dtype float
    :param upper: Upper bound dtype float

    :return: List of random bits with a bit being a ndarray array of 0 and 1.
    """

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


def cauchyrand_bit_pop_IEEE(shape: Union[Iterable, float], bitsize: int, loc: float,
                            scale: float) -> np.ndarray:
    """
    Generate a cauchy distributed bit population with floats converted with
    float2NdbitIEEE754 and NdbittofloatIEEE754.

    :param shape: Population size dtype tuple [individuals, variables]
    :param bitsize: Bitsize dtype int
    :param loc: loc dtype float
    :param scale: scale dtype float

    :return: List of bits that are cauchy distributed with a bit being a ndarray array of 0 and 1.
    """
    if isinstance(shape, int):
        shape = (shape, 1)
    elif len(shape) == 1:
        shape = (shape[0], 1)

    size = shape[0] * shape[1]

    pop_float = cauchy.rvs(loc=loc, scale=scale, size=size)
    pop_float = np.array(np.array_split(pop_float, int(size/shape[0])), dtype=float)

    blist = []
    for val in range(pop_float.shape[1]):
        blist.append(float2NdbitIEEE754(pop_float[:, val], bitsize))

    return np.array(blist)


def uniform_bit_pop_IEEE(shape: Union[Iterable, float], bitsize: int,
                         boundaries: List[int]) -> np.ndarray:
    """
    Generate a uniform distributed bit population with floats converted with
    float2NdbitIEEE754 and NdbittofloatIEEE754.

    :param shape: Population size dtype tuple [individuals, variables]
    :param bitsize: Bitsize dtype int
    :param boundaries: Boundaries dtype list [lower, upper]

    :return: List of random bits with a bit being a ndarray array of 0 and 1.
    """

    if isinstance(shape, int):
        shape = (shape, 1)
    elif len(shape) == 1:
        shape = (shape[0], 1)

    size = shape[0] * shape[1]
    low, high = boundaries

    pop_float = np.random.uniform(low, high, size)
    pop_float = np.array(np.array_split(pop_float, int(size/shape[0])), dtype=float)
    blist = []
    for val in range(pop_float.shape[1]):
        blist.append(float2NdbitIEEE754(pop_float[:, val], bitsize))

    return np.array(blist)


# int2ndbit and ndbit2int routines

def bitpop(shape: list, bitsize: int = 8, factor = 1.0, bias = 0.0):
    """
    Generate a bit population within boundaries imposed by the factor, bias and
    bitsize. Using floats converted with int2ndbit and ndbit2int.

    :param shape: Population size dtype tuple [individuals, variables]
    :param bitsize: Bitsize dtype int
    :param factor: Factor dtype float
    :param bias: Bias dtype float

    :return: List of random bits with a bit being a ndarray array of 0 and 1.
    """

    if isinstance(shape, int):
        shape = (shape, 1)
    elif len(shape) == 1:
        shape = (shape[0], 1)

    shape[1] *= bitsize

    blist = []
    for val in range(shape[0]):
        blist.append(np.random.randint(0, 2, shape[1]))

    return np.array(blist, dtype=np.uint8)

def uniform_bit_pop(shape: Iterable, bitsize: int, boundaries: list,
                    factor: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """
    Generate a uniform bit population with floats converted with
    int2ndbit and ndbit2int.

    :param shape: Population size dtype tuple [individuals, variables]
    :param bitsize: Bitsize dtype int
    :param boundaries: Boundaries [0] lower, [1] upper dtype list of float or int

    :return: List of random bits with a bit being a ndarray array of 0 and 1.
    """

    lower, upper = boundaries

    pop_float = np.vstack(np.array_split(np.random.uniform(lower, upper, shape[0] * shape[1]), shape[0]))

    if factor < np.abs(lower) or factor < np.abs(upper):
        factor = np.abs(lower) if np.abs(lower) > np.abs(upper) else np.abs(upper)

    return int2ndbit(pop_float, bitsize, factor=factor, bias=bias)

def cauchy_bit_pop(shape: Iterable, bitsize: int, loc: float, scale: float,
                   factor: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """
    Generate a cauchy bit population with floats converted with
    int2ndbit and ndbit2int.

    :param shape: Population size dtype tuple [individuals, variables]
    :param bitsize: Bitsize dtype int
    :param loc: loc dtype float
    :param scale: scale dtype float

    :return: List of random bits with a bit being a ndarray array of 0 and 1.
    """
    pop_float = np.vstack(np.array_split(cauchy.rvs(loc=loc, scale=scale, size=shape[0] * shape[1]), shape[0]))

    pop_float = (pop_float / np.abs(pop_float).max()) * factor + bias
    if factor < np.abs(loc):
        factor = np.abs(loc)

    return int2ndbit(pop_float, bitsize, factor=factor, bias=bias)

if __name__ == "__main__":
    print(ndbit2int(cauchy_bit_pop([10, 4], 8, 4, 10, factor=10), bitsize=8, factor=10))