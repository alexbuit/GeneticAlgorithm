
import matplotlib.pyplot as plt
import numpy as np

# read the data
# data_python = np.loadtxt("error_rate_PYTHON.txt", delimiter=",", skiprows=1)
# data_c = np.loadtxt("error_rate_C.txt", delimiter=" ", skiprows=1)
#
# # plot the data
# plt.scatter(data_python[:, 0], data_python[:, 1], label="Python", color="C0")
# plt.plot(data_python[:, 0], data_python[:, 1], color="C0", linestyle="--")
# plt.scatter(data_c[:, 0], data_c[:, 1], label="C", color="C1")
# plt.plot(data_c[:, 0], data_c[:, 1], color="C1", linestyle="--")
#
# # add labels
# plt.xlabel("Bitsize")
# plt.ylabel("Error rate")
#
# # add legend
# plt.legend()
# plt.grid()
#
# # show the plot
# plt.show()
#
# plt.clf()
#
# # do the same for the zoomed in plot
# data_zoom_python = np.loadtxt("error_rate_zoom_PYTHON.txt", delimiter=",", skiprows=1)
# data_zoom_c = np.loadtxt("error_rate_zoom_c.txt", delimiter=" ", skiprows=1)
#
# # plot the data
# plt.scatter(data_zoom_python[:, 0], data_zoom_python[:, 1], label="Python", color="C0")
# plt.plot(data_zoom_python[:, 0], data_zoom_python[:, 1], color="C0", linestyle="--")
# plt.scatter(data_zoom_c[:, 0], data_zoom_c[:, 1], label="C", color="C1")
# plt.plot(data_zoom_c[:, 0], data_zoom_c[:, 1], color="C1", linestyle="--")
#
# # add labels
# plt.xlabel("Bitsize")
# plt.ylabel("Error rate")
#
# # add legend
# plt.legend()
# plt.grid()
#
# # show the plot
# plt.savefig("error_rate_zoom.png")

# Compare conversion speed of python and C
data_conversion_python = np.loadtxt("benchmark_conversion.txt", delimiter=";", skiprows=0).T
data_conversion_c = np.loadtxt("conversion_benchmarks_C.txt", delimiter=";", skiprows=1)

# plot the data for python int2bit
plt.scatter(np.arange(1, 40), data_conversion_python[:, 1], label="int2ndbit", color="C0")
# plot the data for c int2bit
plt.scatter(np.arange(1, 40), data_conversion_python[:, 0], label="ndbit2int", color="C1")

# add title
plt.title("Conversion speed in Python for 16 bit integers in a pop of 16")

# add labels
plt.xlabel("Amount of genes")
plt.ylabel("Time [ms]")

# add legend
plt.legend()
plt.grid()

plt.show()
