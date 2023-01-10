
import numpy as np
import struct

from matplotlib import pyplot as plt
from matplotlib import cm

from src.test_functions import allfx, dim2fx, ndimfx

bdict = {8: [1, 4, 3], 16: [1, 5, 10], 32: [1, 8, 23], 64: [1, 11, 52],
         128: [1, 15, 112], 256: [1, 19, 236]}


def decdicts(func):
    def wrapper(*args):
        return func()[args[0]]
    return wrapper


@decdicts
def floatdict():
    return {8: [1, 4, 3], 16: [1, 5, 10], 32: [1, 8, 23], 64: [1, 11, 52],
         128: [1, 15, 112], 256: [1, 19, 236]}

@decdicts
def bitdict():
    return {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}


def int_to_binary(integer, size):
    binary_arr = np.zeros(shape=size, dtype=np.uint8)
    i = 0
    while(integer > 0):
        digit = integer % 2
        binary_arr[i] = digit
        integer = integer // 2
        i += 1
    # binary_arr = np.flip(binary_arr)
    return binary_arr


def b2int(bit: np.ndarray) -> np.ndarray:
    """
    Conversion of m x n (big endian) bit array to integers.
    :param bit: m x n ndarray of numpy integers (0, 1) representing a bit
    :return: m x n ndarray of integers
    """
    # credits Geoffrey Andersons solution
    # https://stackoverflow.com/questions/41069825/convert-binary-01-numpy-to-integer-or-binary-string
    if len(bit.shape) > 1:
        m, n = bit.shape  # number of columns is needed, not bits.size
    else:
        n = bit.size
    a = 2 ** np.arange(n) # [::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bit @ a  # this matmult is the key line of code


def b2sfloat(bit: np.ndarray) -> np.ndarray:
    """
    Conversion of m x n (big endian) bit array (numpy) to IEEE 754 single precision float
    according to Jaime's solution
    (https://stackoverflow.com/questions/26018448/convert-a-binary-string-into-ieee-754-single-precision-python)
    :param bit: m x n ndarray of numpy integers representing a bit
    :return: m x n ndarray of IEEE 754 single precision floats
    """
    return np.packbits(bit.reshape(-1, 8)).reshape(-1, 4)[:, ::-1].copy().view(
        np.float32).transpose()[0]


def b2dfloat(bit: np.ndarray) -> np.ndarray:
    """
    Conversion of bit m x n (big endian) bit array (numpy) to IEEE 754 double precision float
    according to Jaime's solution
    (https://stackoverflow.com/questions/26018448/convert-a-binary-string-into-ieee-754-single-precision-python)
    :param bit: m x n ndarray of numpy integers representing a bit
    :return: m x n ndarray of IEEE 754 double precision floats
    """
    return np.packbits(bit.reshape(-1, 16)).reshape(-1, 8)[:, ::-1].copy().view(
        np.float64).transpose()[0]


def floatToBinary64(val):
    """
    https://www.technical-recipes.com/2012/converting-between-binary-and-decimal-representations-of-ieee-754-floating-point-numbers-in-c/
    :param value:
    :return:
    """
    value = struct.unpack("Q", struct.pack('d', val))[0]
    if val < 0:
        return np.array([int(i) for i in format(value, "#065b")[2:]])

    if val > 0:
        return np.array([int(i) for i in "0" + format(value, "#065b")[2:]])

    else:
        return np.array([int(i) for i in "".join("0" for _ in range(64))])


def floatToBinary32(val):
    """
    https://www.technical-recipes.com/2012/converting-between-binary-and-decimal-representations-of-ieee-754-floating-point-numbers-in-c/
    :param value:
    :return:
    """
    value = struct.unpack("L", struct.pack('f', val))[0]
    if val < 0:
        return np.array([int(i) for i in format(value, "#033b")[2:]])

    if val > 0:
        return np.array([int(i) for i in "0" + format(value, "#033b")[2:]])

    else:
        return np.array([int(i) for i in "".join("0" for _ in range(32))])


def Ndbit2float(valarr: np.ndarray, bitsize: int, **kwargs) -> np.ndarray:
    global bdict
    b2f = b2dfloat if bitsize == 64 else b2sfloat

    if valarr.ndim == 1:
        valarr = valarr[np.newaxis, :]

    shape = list(valarr.shape)
    shape[-1] = int(np.ceil(shape[-1]/bitsize))

    resmat = np.zeros(shape)

    for b in range(shape[0]):

        pairarr = valarr[b, :]

        nbits = int(pairarr.size/bitsize)

        # Seperate all the sign, exp and mantissa arrays
        sign = pairarr[:nbits * 1][:, np.newaxis]
        exp = np.asarray(np.array_split(pairarr[nbits:nbits * bdict[bitsize][1] + nbits], nbits))
        mant = np.asarray(np.array_split(pairarr[nbits + nbits * bdict[bitsize][1]:], nbits))

        # print(sign, exp, mant)
        # print(sign.shape, exp.shape, mant.shape)

        # Concat the rows of bits together
        res = np.concatenate([sign.T, exp.T, mant.T]).T

        resarr = np.zeros(res.shape[0])
        for row in range(res.shape[0]):
            resarr[row] = b2f(res[row, :])[:, np.newaxis]

        resmat[b, :] = resarr

    # Note if input dim is 1 will return array with dim 2.
    return resmat


def float2Ndbit(valarr: np.ndarray, bitsize: int) -> np.ndarray:
    global bdict

    f2b = floatToBinary64 if bitsize == 64 else floatToBinary32

    valarr = np.array([f2b(val) for val in valarr])

    sign = valarr[:, 0][:, np.newaxis].flatten()
    exp = valarr[:, 1:bdict[bitsize][1] + 1].flatten()
    mantissa = valarr[:, bdict[bitsize][1] + 1:].flatten()

    return np.concatenate([sign, exp, mantissa])


def ndbit2int(valarr: np.ndarray, bitsize: int, normalised: bool = True,
              **kwargs):
    """

    :param valarr: MxN matrix of 0, 1 with dtype np.uint8
                   with M arrays and N the length of val * bitsize
    :param bitsize: Size of the bit, big endian, first bit is sign the others are val
    :param normalised: divide by 2**bitsize-1 and apply factor / bias if true
    :param kwargs: factor: float / bias: float
    :return:
    """


    factor: float = 1.0
    if "factor" in kwargs:
        factor = kwargs["factor"]

    bias: float = 0.0
    if "bias" in kwargs:
        bias = kwargs["bias"]

    if valarr.ndim == 1:
        valarr = valarr[np.newaxis, :]

    shape = list(valarr.shape)
    shape[-1] = int(np.ceil(shape[-1]/bitsize))

    resmat = np.zeros(shape)

    for b in range(shape[0]):

        pairarr = valarr[b, :]

        nbits = int(pairarr.size/bitsize)

        # Split and stack vertically to order have all sign bits in pos 0
        res = np.array_split(pairarr, nbits)
        res = np.vstack(res)

        sign = res[:, 0]

        # If 0 -> 1 if 1 -> -1
        sign = (-1) ** sign

        res = res[:, 1:]

        if normalised:
            # Normalise and apply factors / bias
            resmat[b] = sign * b2int(res)/2**(bitsize - 1) * factor + bias
        else:
            # return a number between -1 * 2 ** bitsize-1 or 1 * 2 ** bitsize-1
            resmat.dtype = np.int64
            resmat[b] = int(sign * b2int(res))

    return resmat


def int2ndbit(valarr: np.ndarray, bitsize: int, normalised: bool = True, **kwargs):
    factor: float = 1.0
    if "factor" in kwargs:
        factor = kwargs["factor"]

    bias: float = 0.0
    if "bias" in kwargs:
        bias = kwargs["bias"]

    valarr = np.array((valarr - bias)/factor * 2**(bitsize - 1), dtype=int)

    shape = list(valarr.shape)
    if len(shape) == 1:
        shape.append(1)

    shape[-1] = int(np.ceil(shape[-1]*bitsize))
    print(shape)


    res = np.zeros(shape, dtype = np.uint8)
    for arr in range(shape[0]):
        bit = np.zeros(shape[1])
        for val in range(valarr.shape[1]):
            # convert int to binary numpy array and add to bit array with sign bit in pos 0
            bit[val * bitsize + 1: (val + 1) * bitsize] = int_to_binary(abs(valarr[arr, val]), bitsize - 1)
            # assign the signed bit, 0 if >0 1 if < 0
            bit[val * bitsize] = (lambda x: 0 if x > 0 else 1)(valarr[arr, val])

        res[arr] = bit

    return res


def convertpop2n(bit2num = None, target = None, **kwargs):
        """
        FROM genetic_algortim.get_numeric() TO BE USED IN OTHER FILES.

        Convert results list with np.ndarrays of dimension mx1 to numeric data
        using provided method or builtin routine for multi variable
        float conversion.
        Provided method needs to accept np.ndarray with binary information
        stored as dtype np.uint8 and return np.ndarray of shape nx1 contaning
        numeric data of float, int.
        Example
        def b2int(bit: np.ndarray) -> np.ndarray:
            '''
            Conversion of m x n (big endian) bit array to integers.
            :param bit2num: m x n ndarray of numpy integers (0, 1) representing a bit
            :return: m x n ndarray of integers
            '''
            m, n = bit.shape
            a = 2 ** np.arange(n)
            return bit @ a
        :param bit2num:
            Binary to numeric conversion routine
        :param kwargs:
            kwargs for bit2num
        :return: numeric values for results of the loaded GA data
        """
        return [bit2num(i, **kwargs) for i in target]


def sigmoid(x):
    return 1 / (1 + pow(np.e, -x))

def sigmoid_derivative(x):
    return x * (1 - x)

def sigmoid2(x, a = 1, b = -1, c = 0.5, d=0 ,Q = 0.5, nu = 1):
    return a + (b - a) / (1 + Q * np.exp(-c * (x - d)))**(1/nu)

def is_decorated(func):
    try:
        return hasattr(func, '__wrapped__') or func.__name__ not in globals()
    except AttributeError:
        return hasattr(func, "__wrapped__") or func.__class__.__name__ not in globals()

def plot3d(fx, min, max, resolution = 100, mode= "plot_surface", **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = np.zeros(shape=(resolution, resolution, 3))

    linsp = np.linspace(min, max, resolution)

    X, Y = np.meshgrid(linsp, linsp)
    Z = fx([X, Y])

    plot = getattr(ax, mode)(X, Y, Z, **kwargs)
    fig.colorbar(plot, shrink=0.5, aspect=5)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x_1, x_2)$")

    ax.title.set_text(f"$f(x_1, x_2) = ${fx.__name__}")

    plt.show()

    return None


if __name__ == "__main__":
    from time import time
    from AdrianPack.Aplot import Default
    from population_initatilisation import *

    from test_functions import *

    def z(x):
        return x[0] + x[1]

    for fx in ndimfx + dim2fx:
        plot3d(fx, -5, 5, resolution= 500, mode= "plot_surface", cmap=cm.coolwarm)

    # start, stop = 0, 100
    # tlist = []
    # tstart = time()
    # low, high, step = 100, 10000, 100
    # test_rng = range(low, high, step)
    #
    # for size in test_rng:
    #     if size % 1000 == 0:
    #         print("%s / %s" % (str(size + 1000), high))
    #
    #     arr = np.linspace(start, stop, size)
    #
    #     barr = float2Ndbit(arr, 64)
    #     farr = Ndbit2float(barr, 64)
    #
    #     np.testing.assert_almost_equal(arr, farr, 4)
    #
    #     tlist.append(time() - tstart)
    #     tstart = time()
    #
    # pl = Default(list(test_rng), tlist, x_label="N-values", y_label="time", degree=1,
    #              decimal_comma=False)
    # print(pl.fit_stats())
    # pl()
    # pl.save_as = "Testbinconvs%sl%sh%s.png" % (low, high, step)
    # pl()




