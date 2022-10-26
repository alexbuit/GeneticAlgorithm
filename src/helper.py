
import numpy as np
import struct

bdict = {8: [1, 4, 3], 16: [1, 5, 10], 32: [1, 8, 23], 64: [1, 11, 52],
         128: [1, 15, 112], 256: [1, 19, 236]}


def bindict():
    return {8: [1, 4, 3], 16: [1, 5, 10], 32: [1, 8, 23], 64: [1, 11, 52],
         128: [1, 15, 112], 256: [1, 19, 236]}


def b2int(bit: np.ndarray) -> np.ndarray:
    """
    Conversion of m x n (big endian) bit array to integers.
    :param bit: m x n ndarray of numpy integers (0, 1) representing a bit
    :return: m x n ndarray of integers
    """
    # credits Geoffrey Andersons solution
    # https://stackoverflow.com/questions/41069825/convert-binary-01-numpy-to-integer-or-binary-string
    m, n = bit.shape  # number of columns is needed, not bits.size
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

    if value > 0:
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

    if value > 0:
        return np.array([int(i) for i in "0" + format(value, "#033b")[2:]])

    else:
        return np.array([int(i) for i in "".join("0" for _ in range(32))])


def Ndbit2float(valarr: np.ndarray, bitsize: int) -> np.ndarray:
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


if __name__ == "__main__":
    from time import time
    from AdrianPack.Aplot import Default
    start, stop = 0, 100
    tlist = []
    tstart = time()
    low, high, step = 100, 10000, 100
    test_rng = range(low, high, step)

    for size in test_rng:
        if size % 1000 == 0:
            print("%s / %s" % (str(size + 1000), high))

        arr = np.linspace(start, stop, size)

        barr = float2Ndbit(arr, 64)
        farr = Ndbit2float(barr, 64)

        np.testing.assert_almost_equal(arr, farr, 4)

        tlist.append(time() - tstart)
        tstart = time()

    pl = Default(list(test_rng), tlist, x_label="N-values", y_label="time", degree=1,
                 decimal_comma=False)
    print(pl.fit_stats())
    pl()
    pl.save_as = "Testbinconvs%sl%sh%s.png" % (low, high, step)
    pl()




