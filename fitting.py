import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from analysis import peak_finder_with_derivatives


def lorentzian(x, gamma, x0, c, a):  # returns a lorentzian evaluated at x (x can be an array)
    return c + a * (0.5 * gamma / (np.pi * (x - x0) ** 2 + (0.5 * gamma) ** 2))


def fit(file_name_and_path):
    data = np.loadtxt(file_name_and_path)
    fpeak = peak_finder_with_derivatives(data)
    x = data[:, 0]
    y = data[:, 1]

    x = x[:int(len(x) * 0.6)]
    y = y[:int(len(y) * 0.6)]

    plt.plot(x, y)
    values, errors = curve_fit(f=lorentzian, xdata=x, ydata=y, bounds=([0, 50, 0, 0], [1e5, 3500, 2, 1e5]))
    print(f"gamma = {values[0]}\nx0 = {values[1]}\nc = {values[2]}\na = {values[3]}")
    print(f"found max (x0) = {fpeak}")
    # print(f"{errors = }")
    plt.plot(x, lorentzian(x, values[0], values[1], values[2], values[3]))
    plt.legend(["Data", "Lorentzian fit"])
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Response RMS / V")
    plt.ylim([0, plt.ylim()[1]])
    plt.show()
