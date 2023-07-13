
import numpy as np

def IEEE_mutate(bit, bitsize, **kwargs) -> np.ndarray:
    """
    Mutate a bit string from mantissa to bitsize

    :param bit: bit array to IEEE_mutate
    :param bitsize: size of the bit array
    :param kwargs: mutate_coeff => number of mutations to apply

    :return: mutated bit array
    """
    global bdict

    bitc = bit.copy()
    nbits = int(bitc.size/bitsize)


    mutate_coeff = int(bit.size/bitsize)
    if "mutate_coeff" in kwargs:
        mutate_coeff = kwargs["mutate_coeff"]

    # mutations = np.random.randint(nbits * (1 + bdict[bitsize][1]), bit.size, mutate_coeff)
    mutations = np.random.choice(np.arange(nbits * (1 + bdict[bitsize][1]), bit.size), mutate_coeff, replace=False)
    # Speed up?
    for mutation in mutations:
        if bitc[mutation]:
            bitc[mutation] = 0
        else:
            bitc[mutation] = 1

    return bitc


def mutate(bit, bitsize, **kwargs) -> np.ndarray:
    """
    Mutate a bit array from 0 to bitsize

    :param bit: bit array to IEEE_mutate
    :param bitsize: size of the bit array
    :param kwargs: mutate_coeff => number of mutations to apply

    :return: mutated bit array
    """
    global bdict

    bitc = bit.copy()

    mutate_coeff = int(bit.size/bitsize)
    if "mutate_coeff" in kwargs:
        mutate_coeff = kwargs["mutate_coeff"]
    # mutations = np.random.randint(nbits * (1 + bdict[bitsize][1]), bit.size, mutate_coeff)
    mutations = np.random.choice(np.arange(0, bit.size), mutate_coeff, replace=False)
    # Speed up?
    for mutation in mutations:
        if bitc[mutation]:
            bitc[mutation] = 0
        else:
            bitc[mutation] = 1

    return bitc