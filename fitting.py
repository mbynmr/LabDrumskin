import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
# from tqdm import tqdm

# from analysis import peak_finder_with_derivatives
# from my_tools import copy2clip


def lorentzian(x, gamma, x0, c, a):
    """
    returns a lorentzian evaluated at x (x can be an array)
    the width of the peak is a function of "gamma"
    the centre of the peak is at position "x0"
    the scale factor "a" controls the height of the curve
    average background noise is compensated for by "c"
    """
    return c + a * (0.5 * gamma / (np.pi * (np.asarray(x) - x0) ** 2 + (0.5 * gamma) ** 2))


def find_peaks():
    order = 200 / 5  # 200Hz wide comparisons
    data = np.loadtxt(r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" +
                      r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\PSY" +
                      r"\2 percent\2023_07_05_15_43_21_P_PSY2_5_20.txt")
    a = argrelextrema(data[:, 1], np.greater, order=int(order))
    print(a)
    print(data[a, 0])
    raise RuntimeError("dun ty")
    # return a


def fit_fast(x, y):
    try:
        values = curve_fit(f=lorentzian, xdata=x, ydata=y, bounds=([0, 50, 0, 0], [1e5, 5e3, 2, 1e5]))[0]
    except RuntimeError:
        # print("Fit problem! Ignoring...")
        values = [1, 1, 1, 1]
    return lorentzian(x, *values), values


def fit(file_name_and_path, copy=True, cutoff=None):
    data = np.loadtxt(file_name_and_path)
    data = data[np.argsort(data, axis=0)[:, 0]]  # sort it!

    if cutoff is not None:
        if 0 <= cutoff[0] < cutoff[1] <= 1:
            data = data[int(len(data[:, 0]) * cutoff[0]):int(len(data[:, 0]) * cutoff[1]), :]
    # fpeak = peak_finder_with_derivatives(data)
    x = data[:, 0]
    y = data[:, 1]

    if copy:
        plt.plot(x, y, label="Data")
    # values = fit_fast(x, y)[1]
    try:
        out = curve_fit(f=lorentzian, xdata=x, ydata=y, bounds=([0, 50, 0, 0], [1e5, 5e3, 2, 1e5]))
    except RuntimeError:
        print("\rcurve fit error, sorry about that", end='')
        out = np.zeros([2, 2])
    # my interpreter is complaining that there are too many values to unpack unless I unpack separately like this
    values, errors = out[0], out[1]
    errors = np.sqrt(np.diag(errors))
    # print(f"{errors = }")
    # print(f"gamma = {values[0]} pm {errors[0]}\nx0 = {values[1]} pm {errors[1]} <----\n"
    #       f"c = {values[2]} pm {errors[2]}\na = {values[3]} pm {errors[3]}")
    if copy:
        try:
            x0str = f"{values[1]:.{int(len(str(values[1]).split('.')[0]) + len(f'{errors[1]:.1g}'.split('.')[1]))}g}"
            # seen this error caused by this above line:
            # TypeError: unsupported format string passed to numpy.ndarray.__format__
        except IndexError:
            x0str = f"{values[1]:.5g}"
        print(f"\nx0 = " + x0str + f"\nx0std = {errors[1]:.1g}")
        print(f"fmax = {x[np.argmax(y)]:.5g}")
        # copy2clip(x0str)
    else:
        return values[1], errors[1]
    fity = lorentzian(x, *values)
    plt.plot(x, fity, label="Lorentzian Fit")
    # half = plt.ylim()[1] / 2
    # plt.plot(x, half + y - fity, label="Difference")
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
    # plt.plot([min(x), max(x)], [half, half], 'k--', label="_Zero line")
    plt.legend()
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Response RMS / V")
    plt.ylim([0, plt.ylim()[1]])
    plt.tight_layout()
    plt.show()


def fit_agg(file_name_and_path, cutoff=None):
    data = np.loadtxt(file_name_and_path)
    data = data[np.argsort(data, axis=0)[:, 0]]  # sort it!

    if cutoff is not None:
        if 0 <= cutoff[0] < cutoff[1] <= 1:
            data = data[int(len(data[:, 0]) * cutoff[0]):int(len(data[:, 0]) * cutoff[1])]
    data = data[np.argwhere(data[:, 1] > np.mean(data[:, 1])), :]  # to do ignore small values it seems... hmmm...
    # data = data[np.argwhere(data[:, 1] > np.mean(data[:, 1])), :]
    x = data[:, 0]
    y = data[:, 1]

    out = curve_fit(f=lorentzian, xdata=x, ydata=y, bounds=([0, 750, 0, 0], [1e5, 5e3, 2, 1e5]))
    values, errors = out[0], out[1]
    errors = np.sqrt(np.diag(errors))
    return values[1], errors[1]


# class FitFailException(Exception):
#     pass
