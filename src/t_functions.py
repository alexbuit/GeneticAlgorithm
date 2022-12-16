
from typing import Union, Callable

import matplotlib.pyplot as plt
import numpy as np

import threading as th

# np.random.seed(12424)



class _tfx_decorator:
    def __init__(self, func: Callable, ndim: int= 1, cores: int =1, compute_analytical: bool = False, **kwargs):
        self.func: Callable = func
        self.cores = cores
        self.ndim = ndim # only necessary for computing minimal and maximal values

        self.samplesize = kwargs.get("sample_size", int(1e5) * cores)

        if compute_analytical:
            # The search area for the optima and minima.
            self.high, self.low = kwargs.get("high", 500), kwargs.get("low", -500)

            # Calculate the optima and minima and represent in dict {x: [x1, x2, ... , xn], fx: []}
            self.optima, self.minima = self.calcoptmin()

        else:
            if func.__name__ in ["wheelers_ridge", "booths_function", "michealewicz", "ackley", "Styblinski_tang"]:
                # minima = {"wheelers_ridge": [], "booths_function", "michealewicz", "ackley", "Styblinski_tang"}
                pass

            else:
                self.optima = kwargs.get("optima", None)
                self.minima = kwargs.get("minima", None)


    def __call__(self, x, *args, **kwargs):
        x = np.asarray(x)

        if x.ndim > 1:
            self.ndim = x.shape[1]
        else:
            self.ndim = x.shape[0]

        return self.func(x, *args, **kwargs)

    def __str__(self):
        return "fx: " +  self.func.__name__ + ", ndim: %s" % self.ndim +\
               ", optima: " + str(self.optima) + ", minima: " + str(self.minima)

    def __repr__(self):
        return self.func.__name__

    def compute(self, x, *args, **kwargs):
        return self.__call__(x, *args, **kwargs)

    def set_dimension(self, ndim):
        self.ndim = ndim
        return None

    def calcoptmin(self):

        # Create a grid of points
        x = np.random.random_sample(self.samplesize)
        x *= (self.high) * np.random.choice([-1, 1], self.samplesize)

        x = x.reshape((int(self.samplesize/self.ndim), self.ndim))

        # Threaded version
        if self.cores > 1:

            opt, mini = [], []


            def calc(j, func):
                k = np.apply_along_axis(func, 1, j)
                opt.append(j[np.argmax(k)])
                mini.append(j[np.argmin(k)])
                return None

            x = np.array_split(x, self.cores)

            threads = []
            for thread in range(0, self.cores):
                threads.append(th.Thread(target=calc, args=(x[thread], self.func)))
                threads[-1].start()

            for thread in threads:
                thread.join()

            optfx = np.apply_along_axis(self.func, 1, np.array(opt))
            minfx = np.apply_along_axis(self.func, 1, np.array(mini))

            print(np.apply_along_axis(self.func, 1, np.array(opt)))

            return {"x": opt[np.argmax(optfx)], "fx": np.max(optfx)}, {"x": mini[np.argmin(minfx)], "fx": np.min(minfx)}

        else:
            x = [x]

            print(x[0].shape)
            opt, mini = [], []
            for j in x:
                k = np.array([self.func(i) for i in j])

                opt = j[np.argmax(k)]
                mini = j[np.argmin(k)]

            optfx = self.func(opt)
            minfx = self.func(mini)

            print(mini)
            return {"x": opt, "fx": np.max(optfx)}, {"x": mini, "fx": np.min(minfx)}


    def gradient(self, x, stepsize: float = 0.01, *args, **kwargs):
        return self.func(x + stepsize, *args, **kwargs)


def tfx_decorator(func: Callable = None, ndim: int= 1, cores: int =1, compute_analytical: bool = True, **kwargs):
    # If a function is given as an argument, return the decorator
    print(func)

    if func is not None:
        return _tfx_decorator(func, ndim, cores, compute_analytical, **kwargs)

    # Otherwise, create a decorator and return it.
    else:
        return lambda f: _tfx_decorator(f, ndim, cores, compute_analytical, **kwargs)

# 1D functions
def tfx(x):
    return 3 * x**2 + 2 * x + 1

## 2D functions
def wheelers_ridge(x: Union[np.ndarray, list], a: float = 1.5) -> float:
    """
    Compute the Wheelersridge function for given x1 and x2
    :param x: list with x1 (otype: float) and x2 (otype: float)
    :param a: additional parameter typically a=1.5
    :return: Value f(x1, x2, a), real float
    """
    x1, x2 = x
    return -np.exp(-(x1 * x2 - a) ** 2 - (x2 - a) ** 2)

def booths_function(x, **kwargs):
    return (x[0] + 2*x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

## N-d functions
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



if __name__ == "__main__":
    from helper import sigmoid,sigmoid2
    # plt.plot(np.linspace(30, 60, 100), [sigmoid2(x, 0, 1, d=50, Q=1, nu=2) for x in np.linspace(30, 60, 100)],
    #          label="$f(x) =  \dfrac{1}{(1 + e^{-1 \cdot (x - 50)})^{1 / 2}}$")
    # plt.xlabel("Power [$mW$]")
    # plt.ylabel("Fitness")

    low, high = -5, 5
    func = tfx_decorator(func=Styblinski_Tang, ndim=40, high=high, low=low, cores=6, sample_size=1000000)

    # x1, x2 = np.linspace(low, high, 1000), np.linspace(low, high, 1000)
    # X1, X2 = np.meshgrid(x1, x2)
    # y = func([X1, X2])

    print(func.minima["x"], func.minima["fx"])
    print(func(np.full(40, -2.903534)))

    # x = np.random.random_sample(int(1e6))
    # x *= (high) * np.random.choice([-1, 1], int(1e6))
    #
    # x = x.reshape((int(1e6 / 2), 2))
    #
    # plt.plot(x[:, 0], x[:, 1], marker="o", linestyle=" ")

    # plt.plot([func.minima["x"][0]], [func.minima["x"][1]], label="Minima", marker="o")
    #
    # plt.pcolormesh(X1, X2, y, cmap='RdBu', shading="auto")
    #
    # plt.xlim(low, high)
    # plt.ylim(low, high)
    #
    # plt.xlabel("$x_1$")
    # plt.ylabel("$x_2$")
    #
    # plt.legend(loc="upper right")
    # plt.colorbar(label="f(x1, x2)")
    #
    # plt.tight_layout()
    #
    # plt.show()

