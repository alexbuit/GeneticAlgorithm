from AdrianPack.Aplot import LivePlot
from AdrianPack.Aplot import Default

import numpy as np
import sys
from time import sleep, time
from random import random

import pyvisa
from ThorlabsPM100 import ThorlabsPM100

from DFM_opt_alg import genetic_algoritm

# from okotech_lib.okodm_sdk.python import okodm_class as oko
#
# handle = oko.open("MMDM 39ch,15mm",
#                   "USB DAC 40ch, 12bit")  # handle for other functions
#
# if handle == 0:
#     sys.exit(("Error opening OKODM device: ", oko.lasterror()))
#
# # Get the number of channels
# n = oko.chan_n(handle)

t_data, v_data = [], []

# def mirror(a):
#     global t_data, v_data
#     sleeptime: float = 1
#
#     # Mirror_type = presumably "MMDM 39ch,15mm"
#     # DAC_type = "USB DAC 40ch, 12bit"
#     # DAC_ids = ^^
#     # oko.open(mirror_type, ) => open the dfm using
#     tstart = time()
#
#     # Set the actuators with array between 0 and 1
#     voltages = np.zeros(n)
#     if not oko.set(handle, voltages):
#         sys.exit("Error writing to OKODM device: " + oko.lasterror())
#
#     sleep(0.5)
#
#     # for i in range(1, n):
#     #     # Set current voltage to 1
#     #     voltages[i] = 1
#     #     # Set previous voltage to 0
#     #     voltages[i - 1] = 0
#     #
#     #     if not oko.set(handle, voltages):
#     #         sys.exit("Error writing to OKODM device: " + oko.lasterror())
#     #
#     #     sleep(sleeptime)
#     #     print(i)
#
#     rng = range(a, n + a, a)
#     for _ in range(0, 5):
#         for i in rng:
#             voltages[i - a:i] = 1
#
#             if not oko.set(handle, voltages):
#                 sys.exit("Error writing to OKODM device: " + oko.lasterror())
#
#             sleep(sleeptime)
#             voltages[i - a:i] = 0
#
#             t_data.append(time() - tstart)
#             v_data.append(voltages)
#
#             sleep(sleeptime)
#
#             voltages[i - a:i] = -1
#
#             t_data.append(time() - tstart)
#             v_data.append(voltages)
#
#             if not oko.set(handle, voltages):
#                 sys.exit("Error writing to OKODM device: " + oko.lasterror())
#
#             sleep(sleeptime)
#
#             voltages[i - a:i] = 0
#
#             t_data.append(time() - tstart)
#             v_data.append(voltages)
#
#             if not oko.set(handle, voltages):
#                 sys.exit("Error writing to OKODM device: " + oko.lasterror())
#
#     voltages = np.zeros(n)
#     if not oko.set(handle, voltages):
#         sys.exit("Error writing to OKODM device: " + oko.lasterror())
#
#     print("finished in t=%s s" % (time() - tstart))
#     oko.close(handle)


rm = pyvisa.ResourceManager()
print(rm.list_resources())
inst = rm.open_resource('USB0::0x1313::0x8078::PM001464::INSTR', timeout=1)

power_meter = ThorlabsPM100(inst=inst)


def print_iets_man():
    global t_data
    for j in range(0, 10):
        sleep(1)
        t_data.append(time())
        print("ik ben bezige bij nr %s" % j)
    return True

print(t_data)


def read():
    return power_meter.read

def GA():
    from t_functions import michealewicz
    print("start")
    end = 200

    tarr = np.zeros(shape=(end, 2))

    for i in np.arange(10, end, 5):
        print("pop size: %s; test until %s" % (i, end))

        tstart = time()

        ga = genetic_algoritm()

        size = [50, 37]
        low = 0
        high = 4

        bitsize = 64

        ga.init_pop("uniform", shape=size, low=low, high=high, bitsize=bitsize)
        ga.target_func(michealewicz)

        ga.run(epochs=i, verbosity=0)

        tarr[i, 0] = time() - tstart
        tarr[i, 1] = i

    with open("gatimetestepoch64.txt", 'w') as f:
        f.write(';'.join([str(i) for i in range(tarr.shape[1])]) + "\n")
        for i in range(tarr.shape[0]):
            f.write(";".join([str(item) for item in tarr[i]]) + "\n")

    # pl = Default(x=tarr[:, 0], y=tarr[:, 1], x_label="size", y_label="time")
    # print(pl.x, pl.y)
    # pl.save_as = r"gatime%s.png" % (1)
    # pl()

    return None

t0 = time()


def t():
    return time() - t0


# fx = mirror
k = 3
fx = GA
earg = 15
lp = LivePlot(x=t, y=read, x_label="time", y_label="Intensity",
              linestyle="")
lp.live_floating_point_average(0, 10)
res = lp.run(interval=1)
# res.save_as = r"lidar%sfx%sarg%s.png" % (k, fx.__name__, earg)
# np.savetxt(r"lidar%sfx%sarg%s.txt" % (k, fx.__name__, earg),
#            np.asarray([res.x, res.y]).T, header="time;Intens",
#            delimiter=";")
# res()
