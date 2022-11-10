
from typing import Union
import numpy as np

def tfx(x):
    return 3 * x**2 + 2 * x + 1


def wheelers_ridge(x: Union[np.ndarray, list], a: float = 1.5) -> float:
    """
    Compute the Wheelersridge function for given x1 and x2
    :param x: list with x1 (otype: float) and x2 (otype: float)
    :param a: additional parameter typically a=1.5
    :return: Value f(x1, x2, a), real float
    """
    x1, x2 = x
    return -np.exp(-(x1 * x2 - a) ** 2 - (x2 - a) ** 2)


def michealewicz(x: list, m: float = 10.0) -> float:
    """
    Compute the Micealewicz function for x1, x2, x...
    :param x: List of x inputs, where N-dimensions = len(x)
    :param m: Steepness parameter, typically m=10
    :return: Value f(x1, x2, ....), real float
    """
    return -sum(
        [np.sin(x[i - 1]) * np.sin((i * x[i - 1] ** 2) / np.pi) ** (2 * m) for i in
         range(1, len(x)+1)])


def ackley(x: list, a: float = 20, b: float = 0.2, c: float = 2 * np.pi):
    """
    Compute Ackley' function for x1, x2, x...
    :param x: list of x inputs where N-dimension = len(x)
    :param a: 
    :param b:
    :param c:
    :return:
    """
    ndim = len(x)
    x = np.array(x, dtype=float)
    return -a * np.exp(-b * np.sqrt(1/ndim * sum(x**2))) - np.exp(1/ndim * sum(np.cos(c * x))) + a + np.exp(1)

