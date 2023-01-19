
from typing import Union, Callable

import matplotlib.pyplot as plt
import numpy as np

import threading as th

# np.random.seed(12424)

from dfmcontrol.test_functions import minima as tf

class _tfx_decorator:
    """
    Decorator class for test functions.
    This class decorates the test functions with the following methods:
    - compute
    - set_dimension
    - dim
    - set_minima
    - set_optima
    - update_optmin
    - gradient

    And the following attributes:
    - func
    - cores
    - ndim
    - samplesize
    - optima
    - minima

    For which it is possible to call the test functions with the same arguments as the original functions.
    The attributes are all initialised with standard values set in the __init__ method. When the function
    is within the list of functions for which the analytical optima and minima are known, the optima and
    minima are calculated and stored in the attributes. Otherwise, the optima and minima are set to None.

    These minima and maxima can either be set manually by calling the set_minima and set_optima methods,
    or they can be calculated by calling the update_optmin method. This method will check if the optima
    and minima are known for given function, if not the optima and minima are calculated by sampling the
    function over a grid of points.
    """
    def __init__(self, func: Callable, ndim: int= 1, cores: int =1, compute_analytical: bool = False, **kwargs):
        self.func: Callable = func
        self.cores = cores
        self.ndim: int = ndim # only necessary for computing minimal and maximal values

        self.samplesize = kwargs.get("sample_size", int(1e5) * cores)

        if compute_analytical:
            # The search area for the optima and minima.
            self.high, self.low = kwargs.get("high", 500), kwargs.get("low", -500)

            # Calculate the optima and minima and represent in dict {x: [x1, x2, ... , xn], fx: []}
            self.optima, self.minima = self.calcoptmin()

        else:
            if func.__name__ in ["wheelers_ridge", "booths_function", "michealewicz", "ackley", "Styblinski_Tang"]:
                self.minima = {"x": getattr(tf, "min" +  func.__name__ + "loc")(self.ndim), "fx": getattr(tf, "min" + func.__name__)(self.ndim)}
                self.optima = {"x": None, "fx": None}
            else:
                self.optima = kwargs.get("optima", {"x": None, "fx": None}) # {"x": [x1, x2, ... , xn], "fx": []}
                self.minima = kwargs.get("minima", {"x": None, "fx": None})


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
        """
        :return: function name
        """
        return self.func.__name__

    def compute(self, x, *args, **kwargs) -> Union[float, np.ndarray]:
        """
        Compute the function value at x.

        :param x: np.ndarray
        :param args: *args
        :param kwargs: **kwargs

        :return: np.ndarray
        """
        return self.__call__(x, *args, **kwargs)

    def set_dimension(self, ndim: int):
        """
        Set the dimension of the function.

        :param ndim: int

        :return: None
        """
        self.ndim = int(ndim)
        self.update_optmin()
        return None

    def dim(self, ndim: int):
        """
        Alias for set_dimension.

        :param ndim: int

        :return: None
        """
        self.set_dimension(ndim)
        return None

    def set_minima(self, **kwargs):
        """
        Set the minima of the function.
        :param kwargs: minima: {"x": [x1, x2, ... , xn], "fx": []}

        :return: None
        """
        self.update_optmin(**kwargs)
        return self.minima

    def set_optima(self, **kwargs):
        """
        Set the optima of the function.

        :param kwargs: optima: {"x": [x1, x2, ... , xn], "fx": []}

        :return: None
        """
        self.update_optmin(**kwargs)
        return self.optima

    def update_optmin(self, **kwargs):
        if self.func.__name__ in ["wheelers_ridge", "booths_function",
                             "michealewicz", "ackley", "Styblinski_Tang"]:
            self.minima = {
                "x": getattr(tf, "min" + self.func.__name__ + "loc")(self.ndim),
                "fx": getattr(tf, "min" + self.func.__name__)(self.ndim)}
            self.optima = {"x": None, "fx": None}
        else:
            self.optima = kwargs.get("optima", {"x": None,
                                                "fx": None})  # {"x": [x1, x2, ... , xn], "fx": []}
            self.minima = kwargs.get("minima", {"x": None, "fx": None})

    # expensive and doesnt work for higher dim functions only recommended for 1d/2d functions
    def calcoptmin(self):
        """
        Calculate the optima and minima of the function by iterating the function of a large grid of points.
        the grid is defined by the high and low attributes.
        This method is unreliable and expensive for higher dimensional functions.

        :return: optima: {"x": [x1, x2, ... , xn], "fx": []}, minima: {"x": [x1, x2, ... , xn], "fx": []}
        """

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

            # print(np.apply_along_axis(self.func, 1, np.array(opt)))

            # return {"x": opt[np.argmax(optfx)], "fx": np.max(optfx)}, {"x": mini[np.argmin(minfx)], "fx": np.min(minfx)}

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

            return {"x": opt, "fx": np.max(optfx)}, {"x": mini, "fx": np.min(minfx)}


    def gradient(self, x, stepsize: float = 0.01, *args, **kwargs):
        """
        Calculate the gradient of the function at x.

        :param x: Input point at which to calculate the gradient.
        :param stepsize: The stepsize to use for the gradient calculation.
        :param args:  function arguments
        :param kwargs: dictionary of function arguments

        :return: The gradient of the function at x.
        """
        return self.func(x + stepsize, *args, **kwargs)


def tfx_decorator(func: Callable = None, ndim: int= 1, cores: int =1, compute_analytical: bool = False, **kwargs):
    """
    Decorator for creating a TFx object.

    :param func: Callable function to be decorated.
    :param ndim: int, dimension of the function.
    :param cores: int, number of cores to use for the calculation of the optima and minima.
    :param compute_analytical: bool, whether to compute the analytical optima and minima.
    :param kwargs: minima: {"x": [x1, x2, ... , xn], "fx": []}, optima: {"x": [x1, x2, ... , xn], "fx": []}

    :return: TFx object
    """
    # If a function is given as an argument, return the decorator
    if func is not None:
        return _tfx_decorator(func, ndim, cores, compute_analytical, **kwargs)

    # Otherwise, create a decorator and return it.
    else:
        dec = lambda f: _tfx_decorator(f, ndim, cores, compute_analytical, **kwargs)

# 1D functions
@tfx_decorator
def tfx(x):
    """
    Test function.

    :param x: Input point.

    :return: The value of the function at x.
    """
    return 3 * x**2 + 2 * x + 1

## 2D functions
@tfx_decorator
def wheelers_ridge(x: Union[np.ndarray, list], a: float = 1.5) -> float:
    """
    Compute the Wheelersridge function for given x1 and x2

    :param x: list with x1 (otype: float) and x2 (otype: float)
    :param a: additional parameter typically a=1.5

    :return: Value f(x1, x2, a), real float
    """
    x1, x2 = x
    return -np.exp(-(x1 * x2 - a) ** 2 - (x2 - a) ** 2)

@tfx_decorator
def booths_function(x: Union[np.ndarray, list]) -> float:
    """
    Compute the Booths function for given x1 and x2

    :param x: list with x1 (otype: float) and x2 (otype: float)

    :return: Value f(x1, x2), real float
    """
    return (x[0] + 2*x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

## N-d functions
@tfx_decorator
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

@tfx_decorator
def ackley(x: list, a: float = 20, b: float = 0.2, c: float = 2):
    """
    Compute Ackley' function for x1, x2, x...

    :param x: list of x inputs where N-dimension = len(x)
    :param a: float, typically a=20
    :param b: float, typically b=0.2
    :param c: float, typically c=2

    :return: Value f(x1, x2, ....), real float
    """
    c *= np.pi  # Needed for autodoc, maybe find a workaround?

    ndim = (lambda i: len(i) if isinstance(i, list) else i.ndim)(x)
    x = np.array(x, dtype=float)
    return -a * np.exp(-b * np.sqrt(1/ndim * sum(x**2))) - np.exp(1/ndim * sum(np.cos(c * x))) + a + np.exp(1)

@tfx_decorator
def Styblinski_Tang(x: list):
    """
    Compute Ackley' function for x1, x2, x...

    :param x: list of x inputs where N-dimension = len(x)

    :return: Value f(x1, x2, ....), real float
    """
    x = np.array(x, dtype=float)
    return sum(x**4) - 16 * sum(x**2) + 5 * sum(x)/2



if __name__ == "__main__":
    from src.helper import sigmoid,sigmoid2
    # plt.plot(np.linspace(30, 60, 100), [sigmoid2(x, 0, 1, d=50, Q=1, nu=2) for x in np.linspace(30, 60, 100)],
    #          label="$f(x) =  \dfrac{1}{(1 + e^{-1 \cdot (x - 50)})^{1 / 2}}$")
    # plt.xlabel("Power [$mW$]")
    # plt.ylabel("Fitness")

    low, high = -5, 5
    func = michealewicz
    func.set_dimension(3)

    # x1, x2 = np.linspace(low, high, 1000), np.linspace(low, high, 1000)
    # X1, X2 = np.meshgrid(x1, x2)
    # y = func([X1, X2])

    print(func)
    print(func.minima["x"], func.minima["fx"])
    # print(func(np.full(40, -2.903534)))

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

