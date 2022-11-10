import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from analysis import peak_finder_with_derivatives


def lorentzian(x, gamma, x0, c, a):
    # returns a lorentzian evaluated at x (x can be an array)
    # the width of the peak is a function of "gamma"
    # the centre of the peak is at position "x0"
    # average background noise is compensated for by "c"
    # the scale factor "a" controls the height of the curve
    return c + a * (0.5 * gamma / (np.pi * (np.asarray(x) - x0) ** 2 + (0.5 * gamma) ** 2))


def fit_fast(x, y):
    try:
        values = curve_fit(f=lorentzian, xdata=x, ydata=y, bounds=([0, 50, 0, 0], [1e4, 5000, 2, 1e4]))[0]
    except RuntimeError:
        print("Fit problem! Ignoring...")
        values = [1, 1, 1, 1]
    return lorentzian(x, values[0], values[1], values[2], values[3]), values


def fit(file_name_and_path, cutoff=None):
    data = np.loadtxt(file_name_and_path)
    data = data[np.argsort(data, axis=0)[:, 0]]  # sort it!

    if cutoff is not None and 0 <= cutoff[0] < cutoff[1] <= 1:
        data = data[int(len(data[:, 0]) * cutoff[0]):int(len(data[:, 0]) * cutoff[1])]
    # fpeak = peak_finder_with_derivatives(data)
    x = data[:, 0]
    y = data[:, 1]

    plt.plot(x, y, label="Data")
    values = fit_fast(x, y)[1]
    # # my interpreter is complaining that there are too many values to unpack unless I unpack separately like this
    # values, errors = out[0], out[1]
    # # print(f"{errors = }")
    print(f"gamma = {values[0]}\nx0 = {values[1]}\nc = {values[2]}\na = {values[3]}")
    # print(f"found max (x0) = {fpeak[0]}")
    fity = lorentzian(x, values[0], values[1], values[2], values[3])
    plt.plot(x, fity, label="Lorentzian Fit")
    # half = (12 * 2 ** (-1 / 2)) / 2
    half = plt.ylim()[1] / 2
    plt.plot(x, half + y - fity, label="Difference")
    # areadiff = np.zeros_like(x)
    # w = 5  # choose a width value
    # for i, ix in enumerate(x):
    #     if i <= w or i >= len(x) - w:
    #         areadiff[i] = 0
    #     else:
    #         areadiff[i] = np.trapz(y=y[i-w:i+w+1], x=x[i-w:i+w+1]) - np.trapz(y=fity[i-w:i+w+1], x=x[i-w:i+w+1])
    # if w != 0:
    #     areadiff = areadiff / (w * (x[1] - x[0]))  # "normalises" to be on a similar scale to normal differences
    # plt.plot(x, half + areadiff, label="Local Area Difference")
    plt.plot([min(x), max(x)], [half, half], 'k--', label="_Zero line")
    plt.legend()
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Response RMS / V")
    plt.ylim([0, plt.ylim()[1]])
    plt.show()
