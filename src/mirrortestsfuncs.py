
import sys
import numpy as np

from time import sleep, time
from AdrianPack.Helper import compress_ind
from okotech_lib.okodm_sdk.python import okodm_class as oko

def mirror(a):
    sleeptime: float = 1

    # Mirror_type = presumably "MMDM 39ch,15mm"
    # DAC_type = "USB DAC 40ch, 12bit"
    # DAC_ids = ^^
    # oko.open(mirror_type, ) => open the dfm using
    tstart = time()
    handle = oko.open("MMDM 39ch,15mm", "USB DAC 40ch, 12bit") # handle for other functions

    if handle == 0:
        sys.exit(("Error opening OKODM device: ", oko.lasterror()))

    # Get the number of channels
    n = oko.chan_n(handle)

    # Set the actuators with array between 0 and 1
    voltages = np.zeros(n)
    if not oko.set(handle, voltages):
        sys.exit("Error writing to OKODM device: " + oko.lasterror())

    sleep(0.5)

    # for i in range(1, n):
    #     # Set current voltage to 1
    #     voltages[i] = 1
    #     # Set previous voltage to 0
    #     voltages[i - 1] = 0
    #
    #     if not oko.set(handle, voltages):
    #         sys.exit("Error writing to OKODM device: " + oko.lasterror())
    #
    #     sleep(sleeptime)
    #     print(i)

    rng = range(a, n + a, a)
    for _ in range(0, 5):
        for i in rng:
            voltages[i - a:i] = 1

            if not oko.set(handle, voltages):
                sys.exit("Error writing to OKODM device: " + oko.lasterror())

            sleep(sleeptime)
            voltages[i - a:i] = 0

    voltages = np.zeros(n)
    if not oko.set(handle, voltages):
        sys.exit("Error writing to OKODM device: " + oko.lasterror())


    print("finished in t=%s s" % (time() - tstart))
    oko.close(handle)

def split_n(n: int, a: int):
    # Array size n, split in a bins with extra
    arr = np.zeros(n)
    print(n % a)
    print(list(range(0, n, a)))
    rng = range(a, n + a, a)
    for i in rng:
        print(arr[i-a:i])
        arr[i-a:i] = 1
        print(arr)
        sleep(1)
        arr[i - a:i] = 0
    # if n % a > 0:
    #     arr[n - n % a:] = 1
    #     print(arr)
    #     sleep(1)
    #     arr[n - n % a:] = 0

    arr[:] = 0
    print(arr)

    return None

split_n(36, 5)