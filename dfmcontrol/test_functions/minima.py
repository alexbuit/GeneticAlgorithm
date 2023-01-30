
import numpy as np

# 1dimension
def mintfx(x: float):
    """
    *need to be implemented*
    This function has a global minimum at x = 0 with a value of 0.

    :param x: Value of the function

    :return: Minimum value of the function at x
    """
    return 0

def mintfxloc(x: float):
    """
    Global minimum for tfx function.

    :param x: Value of the function

    :return: Global minimum location
    """
    return 0

# 2dimension
# cite simone fraser university
def minbooths_function(n: int = 2):
    """
    This function has a global minimum at x = (1, 3) with a value of 0.
    And is only defined for 2 dimensions.

    :param n: Number of dimensions (fixed at 2)

    :return: Minimum value of the function at dimension n
    """
    return 0

def minbooths_functionloc(n: int = 2):
    """
    Global minimum for booths function.

    :param n: Number of dimensions (fixed at 2)

    :return: Global minimum location
    """
    return np.array([1, 3])

# cite optimizers
def minwheelers_ridge(n: int = 2):
    """
    This function has a global minimum at x = (1, 3/2) with a value of  -1.
    And is only defined for 2 dimensions.

    :param n: Number of dimensions (fixed at 2)

    :return: Minimum value of the function at dimension n
    """

    return -1

def minwheelers_ridgeloc(n: int = 2):
    """
    Global minimum for wheelers ridge function.

    :param n: Number of dimensions (fixed at 2)

    :return: Global minimum location
    """

    return np.array([1, 3/2])


# ndimension
# cite "Certified global minima for a benchmark of difficulr optimazation problems"
def minmichealewicz(n: int):
    """
    The global minima for this function can be approximated for n > 10 by the following formula:
    f(x) = -0.99864n + 0.30271
    for dimensions 2 < n < 10 the global minima are defined in a lookup table.

    :param n: Number of dimensions

    :return: Minimum value of the function at dimension n
    """
    if n > 10 or n < 2:
        return n*-0.99864 + 0.30271
    else:
        n = n-1
        return np.array([-1.8013034, -2.7603947, -3.6988571, -4.6876582, -5.6876582, -6.6808853, -7.6637574, -8.6601517, -9.6601517])[n]

def minmichealewiczloc(n: int):
    """
    The global minima for this function are defined between 1 <= n < 10 in a lookup table.

    :param n: Number of dimensions

    :return: Global minimum location
    """
    d = {1: [2.202906]}
    d[2] = d[1] + [1.570796]
    d[3] = d[2] + [1.284992]
    d[4] = d[3] + [1.923058]
    d[5] = d[4] + [1.720470]
    d[6] = d[5] + [1.570796]
    d[7] = d[6] + [1.454414]
    d[8] = d[7] + [1.756087]
    d[9] = d[8] + [1.655717]
    d[10] = d[9] + [1.570796]
    if n <= 10:
        return np.array(d[n])
    else:
        return np.full(n, np.pi/2, float)


# cite simon fraser university
def minackley(n: int):
    """
    This function has a global minimum at x = (0, 0, ..., 0) with a value of 0.

    :param n: Number of dimensions

    :return: Minimum value of the function at dimension n
    """
    return 0

def minackleyloc(n: int):
    """
    Global minimum for Ackleys function.

    :param n: Number of dimensions

    :return: Global minimum location
    """
    return np.zeros(n)

# cite simon fraser university
def minStyblinski_Tang(n: int):
    """
    This function has a global minimum at x = (-2.903534, -2.903534, ..., -2.903534) with a value of -39.16616570377142 * n.

    :param n: Number of dimensions

    :return: Minimum value of the function at dimension n
    """
    return -39.16616570377142 * n

def minStyblinski_Tangloc(n: int):
    """
    Global minimum for Styblinski-Tang function.

    :param n: Number of dimensions

    :return: Global minimum location
    """
    return np.full(n, -2.903534, float)
