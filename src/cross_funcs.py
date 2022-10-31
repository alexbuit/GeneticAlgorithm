import functools

import numpy as np

bdict = {8: [1, 4, 3], 16: [1, 5, 10], 32: [1, 8, 23], 64: [1, 11, 52],
         128: [1, 15, 112], 256: [1, 19, 236]}


def double_point(parent1, parent2, bitsize):
    global bdict

    child1 = np.zeros(parent1.shape, dtype=np.uint8)
    child2 = np.zeros(parent1.shape, dtype=np.uint8)

    nbits = int(parent1.size/bitsize)

    p1_1 = parent1[:nbits * (1 + bdict[bitsize][1])]
    p2_1 = parent2[:nbits * (1 + bdict[bitsize][1])]

    if np.random.randint(0, 1):
        child1[:nbits * (1 + bdict[bitsize][1])] = p1_1[:nbits * (1 + bdict[bitsize][1])]
        child2[:nbits * (1 + bdict[bitsize][1])] = p2_1[:nbits * (1 + bdict[bitsize][1])]
    else:
        child1[:nbits * (1 + bdict[bitsize][1])] = p2_1[:nbits * (1 + bdict[bitsize][1])]
        child2[:nbits * (1 + bdict[bitsize][1])] = p1_1[:nbits * (1 + bdict[bitsize][1])]

    c1 = np.random.randint(nbits * (1 + bdict[bitsize][1]), parent1.size - 1)
    c2 = np.random.randint(c1 + 1, parent1.size)

    child1[nbits * (1 + bdict[bitsize][1]):c1] = parent1[nbits * (1 + bdict[bitsize][1]):c1]
    child1[c1:c2] = parent2[c1:c2]
    child1[c2:] = parent1[c2:]

    child2[nbits * (1 + bdict[bitsize][1]):c1] = parent2[nbits * (1 + bdict[bitsize][1]):c1]
    child2[c1:c2] = parent1[c1:c2]
    child2[c2:] = parent2[c2:]

    return [child1, child2]


def signle_point(parent1, parent2, bitsize):
    global bdict

    child1 = np.zeros(parent1.shape, dtype=np.uint8)
    child2 = np.zeros(parent1.shape, dtype=np.uint8)

    nbits = int(parent1.size/bitsize)

    p1_1 = parent1[:nbits * (1 + bdict[bitsize][1])]
    p2_1 = parent2[:nbits * (1 + bdict[bitsize][1])]

    if np.random.randint(0, 1):
        child1[:nbits * (1 + bdict[bitsize][1])] = p1_1[:nbits * (1 + bdict[bitsize][1])]
        child2[:nbits * (1 + bdict[bitsize][1])] = p2_1[:nbits * (1 + bdict[bitsize][1])]
    else:
        child1[:nbits * (1 + bdict[bitsize][1])] = p2_1[:nbits * (1 + bdict[bitsize][1])]
        child2[:nbits * (1 + bdict[bitsize][1])] = p1_1[:nbits * (1 + bdict[bitsize][1])]

    c1 = np.random.randint(nbits * (1 + bdict[bitsize][1]), parent1.size)
    c2 = np.random.randint(nbits * (1 + bdict[bitsize][1]), parent1.size)

    child1[nbits * (1 + bdict[bitsize][1]):c1] = parent1[nbits * (1 + bdict[bitsize][1]):c1]
    child1[c1:] = parent2[c1:]

    child2[nbits * (1 + bdict[bitsize][1]):c2] = parent1[nbits * (1 + bdict[bitsize][1]):c2]
    child2[c2:] = parent2[c2:]

    return [child1, child2]

def equal_prob_cross(parent1, parent2, bitsize):
    global bdict

    child1 = np.zeros(parent1.shape, dtype=np.uint8)
    child2 = np.zeros(parent1.shape, dtype=np.uint8)

    nbits = int(parent1.size/bitsize)

    p1_1 = parent1[:nbits * (1 + bdict[bitsize][1])]
    p2_1 = parent2[:nbits * (1 + bdict[bitsize][1])]

    if np.random.randint(0, 1):
        child1[:nbits * (1 + bdict[bitsize][1])] = p1_1[:nbits * (1 + bdict[bitsize][1])]
        child2[:nbits * (1 + bdict[bitsize][1])] = p2_1[:nbits * (1 + bdict[bitsize][1])]
    else:
        child1[:nbits * (1 + bdict[bitsize][1])] = p2_1[:nbits * (1 + bdict[bitsize][1])]
        child2[:nbits * (1 + bdict[bitsize][1])] = p1_1[:nbits * (1 + bdict[bitsize][1])]

    for i in range(nbits * (1 + bdict[bitsize][1]), parent1.size):
        if np.random.randint(0, 1):
            child1[i] = parent2[i]
            child2[i] = parent1[i]
        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    return [child1, child2]
