import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from analysis import peak_finder_with_derivatives


def lorentzian(x, gamma, x0, c, a):
    # returns a lorentzian evaluated at x (x can be an array)
    # the width of the peak is a function of "gamma"
    # the centre of the peak is at position "x0"
    # average background noise is compensated for by "c"
    # the scale factor "a" controls the height of the curve
    return c + a * (0.5 * gamma / (np.pi * (np.asarray(x) - x0) ** 2 + (0.5 * gamma) ** 2))


def fit(file_name_and_path, cutoff=None):
    data = np.loadtxt(file_name_and_path)

    if cutoff is not None:
        if 0 <= cutoff[0] < cutoff[1] <= 1:
            data = data[int(len(data[:, 0]) * cutoff[0]):int(len(data[:, 0]) * cutoff[1])]
        else:
            raise Exception("Cutoff is the portion of data cut off from the [start, end] of data."
                            "Values should meet the condition (0 <= cutoff[0] < cutoff[1] <= 1)")

    fpeak = peak_finder_with_derivatives(data)
    x = data[:, 0]
    y = data[:, 1]

    plt.plot(x, y)
    out = curve_fit(f=lorentzian, xdata=x, ydata=y, bounds=([0, 50, 0, 0], [1e5, 3500, 2, 1e5]))
    values = out[0]
    errors = out[1]
    # print(f"{errors = }")
    print(f"gamma = {values[0]}\nx0 = {values[1]}\nc = {values[2]}\na = {values[3]}")
    print(f"found max (x0) = {fpeak[0]}")
    plt.plot(x, lorentzian(x, values[0], values[1], values[2], values[3]))
    plt.legend(["Data", "Lorentzian fit"])
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Response RMS / V")
    plt.ylim([0, plt.ylim()[1]])
    plt.show()
