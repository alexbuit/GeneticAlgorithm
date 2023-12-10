
import matplotlib.pyplot as plt
import numpy as np

# read the data
data_python = np.loadtxt("error_rate_PYTHON.txt", delimiter=",", skiprows=1)
data_c = np.loadtxt("error_rate_C.txt", delimiter=" ", skiprows=1)

# plot the data
plt.scatter(data_python[:, 0], data_python[:, 1], label="Python", color="C0")
plt.plot(data_python[:, 0], data_python[:, 1], color="C0", linestyle="--")
plt.scatter(data_c[:, 0], data_c[:, 1], label="C", color="C1")
plt.plot(data_c[:, 0], data_c[:, 1], color="C1", linestyle="--")

# add labels
plt.xlabel("Bitsize")
plt.ylabel("Error rate")

# add legend
plt.legend()
plt.grid()

# show the plot
plt.show()

plt.clf()

# do the same for the zoomed in plot
data_zoom_python = np.loadtxt("error_rate_zoom_PYTHON.txt", delimiter=",", skiprows=1)
data_zoom_c = np.loadtxt("error_rate_zoom_c.txt", delimiter=" ", skiprows=1)

# plot the data
plt.scatter(data_zoom_python[:, 0], data_zoom_python[:, 1], label="Python", color="C0")
plt.plot(data_zoom_python[:, 0], data_zoom_python[:, 1], color="C0", linestyle="--")
plt.scatter(data_zoom_c[:, 0], data_zoom_c[:, 1], label="C", color="C1")
plt.plot(data_zoom_c[:, 0], data_zoom_c[:, 1], color="C1", linestyle="--")

# add labels
plt.xlabel("Bitsize")
plt.ylabel("Error rate")

# add legend
plt.legend()
plt.grid()

# show the plot
plt.savefig("error_rate_zoom.png")
