from typing import Callable

import numpy as np
import struct

from matplotlib import pyplot as plt
from matplotlib import cm

# from Mathematical_functions import allfx, dim2fx, ndimfx

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


def int_to_binary(integer: int, size: int) -> np.ndarray:
    """
    https://stackoverflow.com/questions/699866/python-int-to-binary

    :param integer: integer to be converted
    :param size: size of binary representation

    :return: binary representation of integer
    """
    binary_arr = np.zeros(shape=size, dtype=np.uint8)
    i = 0
    try:
        while(integer > 0):
            digit = integer % 2
            binary_arr[i] = digit
            integer = integer // 2
            i += 1
    except IndexError:
        binary_arr = np.full(shape=size, fill_value=1, dtype=np.uint8)
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


def floatToBinary64(val: np.float64):
    """
    Conversion of float to binary representation

    https://www.technical-recipes.com/2012/converting-between-binary-and-decimal-representations-of-ieee-754-floating-point-numbers-in-c/
    :param value: float
    :return: binary representation of float
    """
    value = struct.unpack("Q", struct.pack('d', val))[0]
    if val < 0:
        return np.array([int(i) for i in format(value, "#065b")[2:]])

    if val > 0:
        return np.array([int(i) for i in "0" + format(value, "#065b")[2:]])

    else:
        return np.array([int(i) for i in "".join("0" for _ in range(64))])


def floatToBinary32(val: np.float32):
    """
    https://www.technical-recipes.com/2012/converting-between-binary-and-decimal-representations-of-ieee-754-floating-point-numbers-in-c/
    :param value: float
    :return: binary representation of float
    """
    value = struct.unpack("L", struct.pack('f', val))[0]
    if val < 0:
        return np.array([int(i) for i in format(value, "#033b")[2:]])

    if val > 0:
        return np.array([int(i) for i in "0" + format(value, "#033b")[2:]])

    else:
        return np.array([int(i) for i in "".join("0" for _ in range(32))])


def Ndbit2floatIEEE754(valarr: np.ndarray, bitsize: int, **kwargs) -> np.ndarray:
    """
    Conversion of bit m x n (big endian) bit array (numpy) to IEEE 754 double precision float

    :param valarr: m x n ndarray of numpy integers representing a bit
    :param bitsize: Size of the bit, big endian
    :param kwargs:
    :return: m x n/bitsize ndarray of IEEE 754 double precision floats
    """
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


def float2NdbitIEEE754(valarr: np.ndarray, bitsize: int) -> np.ndarray:
    """
    Conversion of bit m x n (big endian) bit array (numpy) to IEEE 754 double precision float
    :param valarr:  m x n ndarray of numpy integers representing a bit
    :param bitsize: Size of the bit, big endian
    :return:  m x n/bitsize ndarray of IEEE 754 double precision floats
    """
    global bdict

    f2b = floatToBinary64 if bitsize == 64 else floatToBinary32

    bitlist = []

    if valarr.ndim == 1:
        valarr = valarr[np.newaxis, :]

    for individual in valarr:
        barr = np.array([f2b(val) for val in individual])

        sign = barr[:, 0][:, np.newaxis].flatten()
        exp = barr[:, 1:bdict[bitsize][1] + 1].flatten()
        mantissa = barr[:, bdict[bitsize][1] + 1:].flatten()

        bitlist.append(np.concatenate([sign, exp, mantissa]))

    if len(bitlist) == 1:
        return bitlist[0]
    else:
        return np.asarray(bitlist)

def ndbit2int(valarr: np.ndarray, bitsize: int, normalised: bool = True,
              **kwargs):
    """
    Conversion of bit m x n (big endian) bit array (numpy) to integer or float
    depending on the normalised flag. If normalised is true, the integer is
    normalised to the range 0 to 1 and a factor / bias may be applied if
    specified in kwargs.

    The factor / bias is applied to the integer after normalisation so that
    the conversion is done as follows

    float = (b2n(valarr)/2^(bitsize) * factor) + bias

    :param valarr: MxN matrix of 0, 1 with dtype np.uint8
                   with M arrays and N the length of val * bitsize
    :param bitsize: Size of the bit, big endian, first bit is sign the others are val
    :param normalised: divide by 2**bitsize-1 and apply factor / bias if true
    :param kwargs: factor: float / bias: float

    :return: MxN matrix of integers or floats (np.float64)
    """
    factor = kwargs.get("factor", 1)
    bias = kwargs.get("bias", 0)

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


def int2ndbit(valarr: np.ndarray, bitsize: int, **kwargs):
    """
    Convert an array of integers (or floats) to a bit array of size bitsize * len(valarr)
    if the valarr is a float, the float will be normalised to the range 0 to 1
    and a factor / bias may be applied if specified in kwargs. So that

    float = valarr / factor - bias

    The float will then be assigned an integer value between 0 and 2**bitsize-1

    int = int(float * 2**bitsize-1)  # Rounding error may occur

    The integer will then be converted to a bit array of size bitsize.

    :param valarr: MxN matrix of integers
    :param bitsize: Size of the bit, big endian, first bit is sign the others are val
    :param kwargs: factor: float / bias: float

    :return: MxN matrix of 0, 1 with dtype np.uint8
    """
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

    res = np.zeros(shape, dtype = np.uint8)
    for arr in range(shape[0]):
        bit = np.zeros(shape[1])
        for val in range(valarr.shape[1]):
            # convert int to binary numpy array and add to bit array with sign bit in pos 0
            bit[val * bitsize + 1: (val + 1) * bitsize] = int_to_binary(abs(valarr[arr, val]), bitsize - 1)
            # assign the signed bit, 0 if > 0, 1 if < 0
            bit[val * bitsize] = (lambda x: 0 if x > 0 else 1)(valarr[arr, val])

        res[arr] = bit

    return res


def convertpop2n(bit2num=None, target=None, bitsize=None, **kwargs):
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
        return [bit2num(i, bitsize, **kwargs) for i in target]


def sigmoid(x):
    """
    Sigmoid function

    :param x: input
    :return: sigmoid(x)
    """
    return 1 / (1 + pow(np.e, -x))

def sigmoid_derivative(x):
    """
    Derivative of sigmoid function

    :param x: input

    :return: sigmoid_derivative(x)
    """
    return x * (1 - x)

def sigmoid2(x, a = 1, b = -1, c = 0.5, d=0 ,Q = 0.5, nu = 1):
    """
    Sigmoid function

    :param x: input
    :param a:
    :param b:
    :param c:
    :param d:
    :param Q:
    :param nu:
    :return: sigmoid(x)
    """
    return a + (b - a) / (1 + Q * np.exp(-c * (x - d)))**(1/nu)

def is_decorated(func: Callable) -> bool:
    """
    Check if function is decorated

    :param func: function of type function

    :return: True if decorated, False if not
    """
    try:
        return hasattr(func, '__wrapped__') or func.__name__ not in globals()
    except AttributeError:
        return hasattr(func, "__wrapped__") or func.__class__.__name__ not in globals()


def flip_a_coin() -> bool:
    """ Flip a coin function with 50/50 chance for True/False"""
    return True if np.random.randint(0, 2) else False


def plot3d(fx, min, max, resolution = 100, mode= "plot_surface", **kwargs):
    """
    Plot 3d function

    :param fx: function to plot
    :param min: min value of x and y
    :param max: max value of x and y
    :param resolution: resolution of the plot
    :param mode: plot_surface or contour

    :param kwargs: kwargs for plot_surface, contour / show = True

    :return: None if show = True, else fig, ax
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = np.zeros(shape=(resolution, resolution, 3))

    linsp = np.linspace(min, max, resolution)

    X, Y = np.meshgrid(linsp, linsp)
    Z = fx([X, Y])
    
    # Instansiate kwargs
    show = kwargs.get("show", True)

    try:
        title = kwargs.get("title", f"$f(x_1, x_2) = ${fx.__name__}")
    except AttributeError:
        title = kwargs.get("title", f"$f(x_1, x_2) = ${fx.__class__.__name__}")

    # Get the plot kwargs
    plargs = kwargs.copy()
    plargs.pop("show", None)
    plargs.pop("title", None)

    # plot
    plot = getattr(ax, mode)(X, Y, Z, **plargs)
    fig.colorbar(plot, shrink=0.5, aspect=5)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x_1, x_2)$")

    ax.title.set_text(title)

    if show:
        plt.show()
        return None

    else:
        return fig, ax

# if __name__ == "__main__":
#     from time import time
#     from AdrianPack.Aplot import Default
#     from population_initatilisation import *
#
#     from Mathematical_functions import *
#
#     def z(x):
#         return x[0] + x[1]
#
#     for fx in ndimfx + dim2fx:
#         plot3d(fx, -5, 5, resolution= 500, mode= "plot_surface", cmap=cm.coolwarm)

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




