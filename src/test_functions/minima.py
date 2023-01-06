
import numpy as np

# 1dimension
def mintfx(x: float):
    return None

# 2dimension
# cite simone fraser university
def minbooths_function(n: int = 2):
    return 0

def minbooths_functionloc(n: int = 2):
    return np.array([1, 3])

# cite optimizers
def minwheelersridge(n: int = 2):
    return -1

def minwheelers_ridgeloc(n: int = 2):
    return np.array([1, 3/2])


# ndimension
# cite "Certified global minima for a benchmark of difficulr optimazation problems"
def minmichealewicz(n: int):
    if n > 10 or n < 2:
        return n*-0.99864 + 0.30271
    else:
        n = n -2
        return np.array([-1.8013034, -2.7603947, -3.6988571, -4.6876582, -5.6876582, -6.6808853, -7.6637574, -8.6601517, -9.6601517])[n]

def minmichealewiczloc(n: int):
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
    return np.array(d[n])


# cite simon fraser university
def minackley(n: int):
    return 0

def minackleyloc(n: int):
    return np.zeros(n)

# cite simon fraser university
def minStyblinski_Tang(n: int):
    return -39.16616570377142 * n

def minStyblinski_Tangloc(n: int):
    return np.full(n, -2.903534)


