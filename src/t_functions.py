
from typing import Union

import matplotlib.pyplot as plt
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


def Styblinski_Tang(x: list):
    """
    Compute Ackley' function for x1, x2, x...
    :param x: list of x inputs where N-dimension = len(x)
    :return:
    """
    x = np.array(x, dtype=float)
    return sum(x**4) - 16 * sum(x**2) + 5 * sum(x)/2

def booths_function(x, **kwargs):
    return (x[0] + 2*x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2


if __name__ == "__main__":
    from helper import sigmoid,sigmoid2
    # plt.plot(np.linspace(30, 60, 100), [sigmoid2(x, 0, 1, d=50, Q=1, nu=2) for x in np.linspace(30, 60, 100)],
    #          label="$f(x) =  \dfrac{1}{(1 + e^{-1 \cdot (x - 50)})^{1 / 2}}$")
    # plt.xlabel("Power [$mW$]")
    # plt.ylabel("Fitness")

    low, high = -5, 5
    func = ackley

    x1, x2 = np.linspace(low, high, 1000), np.linspace(low, high, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    y = func([X1, X2])

    plt.pcolormesh(X1, X2, y, cmap='RdBu', shading="auto")

    plt.xlim(low, high)
    plt.ylim(low, high)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.legend(loc="upper right")
    plt.colorbar(label="f(x1, x2)")

    plt.tight_layout()

    plt.show()

