import matplotlib.pyplot as plt
import numpy as np

import time as t

from dfmcontrol.Helper import *
from dfmcontrol.Utility import crossover, pop, selection

def cross(y, method):
    def fitness(x):
        return sum(x)

    test_pop = pop.uniform_bit_pop([16, y], 16, [0, 1])
    parents, _, _, _ = selection.rank_selection(test_pop, fitness, 16, ndbit2int, b2nkwargs={"factor": 1})

    children = []

    tstart = t.time()

    for p in parents:
        c1, c2 = method(test_pop[p[0]], test_pop[p[1]], bitsize=16)
        children.append(c1)
        children.append(c2)


    return np.asarray(convertpop2n(ndbit2int, children, 16)) - np.asarray(convertpop2n(ndbit2int, test_pop, 16)), t.time() - tstart

x = list(range(1, 40))

single_results_list = []
double_results_list = []
equal_results_list = []

start = t.time()

runtime_single = [start,]
runtime_double = [start,]
runtime_equal = [start,]


for param in x:
    print(param)
    t_results_list1 = []
    t_results_list2 = []
    t_results_list3 = []

    t_runtime_single = [0]
    t_runtime_double = [0]
    t_runtime_equal = [0]

    for i in range(100):
        res = cross(param, crossover.single_point)
        t_results_list1.append(res[0])
        t_runtime_single.append(res[1])
        res = cross(param, crossover.double_point)
        t_results_list2.append(res[0])
        t_runtime_double.append(res[1])
        res = cross(param, crossover.equal_prob)
        t_results_list3.append(res[0])
        t_runtime_equal.append(res[1])

    t_runtime_single = np.diff(t_runtime_single)
    t_runtime_double = np.diff(t_runtime_double)
    t_runtime_equal = np.diff(t_runtime_equal)

    runtime_single.append(np.average(t_runtime_single) * 1000) # convert to ms
    runtime_double.append(np.average(t_runtime_double) * 1000)
    runtime_equal.append(np.average(t_runtime_equal) * 1000)

    zeros1 = np.count_nonzero(np.asarray(t_results_list1) == 0)
    zeros2 = np.count_nonzero(np.asarray(t_results_list2) == 0)
    zeros3 = np.count_nonzero(np.asarray(t_results_list3) == 0)
    single_results_list.append(zeros1)
    double_results_list.append(zeros2)
    equal_results_list.append(zeros3)


plt.scatter(x, single_results_list
            , label="Single Point Crossover")
plt.scatter(x, double_results_list
            , label="Double Point Crossover")
plt.scatter(x, equal_results_list
            , label="Equal Probability Crossover")
plt.title("Crossover Comparison for 16 individuals in a population")
plt.xlabel("Number of genes in an individual")
plt.ylabel("Number of times the crossover resulted in the same individual")
plt.legend()
plt.show()

plt.clf()
plt.scatter(x, runtime_single[1:]
            , label="Single Point Crossover")
plt.scatter(x, runtime_double[1:]
            , label="Double Point Crossover")
plt.scatter(x, runtime_equal[1:]
            , label="Equal Probability Crossover")
plt.title("Crossover Comparison for 16 individuals in a population")
plt.xlabel("Number of genes in an individual")
plt.ylabel("Average runtime of crossover [ms]")
plt.legend()
plt.show()
# save data of time to txt

# concatenate data to one matrix
datamat = np.asarray([x, runtime_single[1:], runtime_double[1:], runtime_equal[1:]]).T

# save data to txt
np.savetxt("crossover_benchmarks.txt", datamat, delimiter=" ", fmt="%1.4f")
