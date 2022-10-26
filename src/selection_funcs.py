
import numpy as np
from helper import *

def roulette_select(pop, fx, bitsize):

    y = np.apply_along_axis(fx, 1, Ndbit2float(pop, bitsize))
    y = np.max(y) - y
    yc = y.copy()
    yrng = np.asarray(range(y.size))
    p = y / sum(y)

    pind = []
    for i in range(int(y.size / 2)):
        if p.size > 1:
            try:
                par = np.random.choice(yrng, 2, p=p, replace=False)
            except ValueError:
                p = np.full(p.size, 1 / p.size)
                par = np.random.choice(yrng, 2, p=p, replace=False)

            pind.append(list(sorted(par).__reversed__()))

            yc = np.delete(yc, np.where(yrng == pind[-1][0])[0][0])
            yc = np.delete(yc, np.where(yrng == pind[-1][1])[0][0])
            yrng = np.delete(yrng, np.where(yrng == pind[-1][0])[0][0])
            yrng = np.delete(yrng, np.where(yrng == pind[-1][1])[0][0])
            p = yc / sum(yc)
    return pind


# def t_roulette_sel(tsize=int(1e6), bitsize=4):
#     """
#
#     :param tsize: Size of the population
#     :param bitsize: Size of a parent within the population
#
#     Correct % => If > 100% or <100% there is are double indexes in the list.
#     :return:
#     """
#     tsart = time()
#     rpop = rand_bit_pop(tsize, bitsize)
#     # print(rpop)
#     parent_list = roulette_select(b2int(rpop), tfx)
#
#     tl = []
#     for parent in parent_list:
#         tl.append(parent[0])
#         tl.append(parent[1])
#
#     # print(len(tl), ":", len(set(tl)))
#     corrperc = 100 - ((len(tl) - len(set(tl))) / (tsize / 2)) * 100
#
#     t = time() - tsart
#     return t, corrperc