import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
import os
import sys
from tqdm import tqdm


def wireplot_manager(path, mode=None, printer=None, ax=None):
    # takes a folder path and extracts the data from every file, plotting it on 3D axes

    if mode is None:
        mode = 'conc'
    if printer is None:
        printer = sys.stderr
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        match mode:
            case 'conc':  # concentration mode
                xstring = 'Concentration / %wt'
            case 'thick':  # thickness mode
                xstring = 'Thickness / nm'
            case 'temp':  # temperature mode
                xstring = 'Temperature / C'
            case 'time':  # time mode
                xstring = 'Time / s'
            case _:
                xstring = 'x axis label'
        ax.set_xlabel(xstring)
        ax.set_ylabel('Frequency / kHz')
        ax.set_zlabel('ln(Normalised Response)')

    files = os.listdir(path)
    files.sort()
    # for i, file in tqdm(enumerate(files), total=len(files), file=printer, ncols=52):
    for i, file in enumerate(files):
        # 2024_03_12_13_11_10_S1V_PSY1_6_20.txt
        if file.split('.')[-1] == 'txt' and len(file.split('_')) >= 10:  # this is specific file name format!!
            x, f, response = data_extractor(path, file, mode)
            x = np.sqrt(110 - x)  # todo sqrt(Tg - T)
            # normalise
            z = response / np.mean(response)
            z = np.log(z)
            # # alternative to ln(z)?
            # ax.set_zlim([0, 4])
            ax.set_ylim([0, 4])
            wireplot(ax, x, f * 1e-3, z)
    # ax.set_ylim([0, 5000])
    plt.show()


def data_extractor(path, file, mode):
    data = np.loadtxt(path + "/" + file)
    f = data[:, 0]
    response = data[:, 1]
    x = np.ones_like(f)
    match mode:
        case 'conc':  # concentration mode
            x = float(file.split('_')[7][3:]) * x
        case 'thick':  # thickness mode
            lookup_table = 1  # todo make a lookup table which turns concentration into thickness (requires measurement)
            conc = float(file.split('_')[7][3:])
            x = lookup_table * x
        case 'temp':  # temperature mode
            x = float(file.split('_')[-1].split('.txt')[0]) * x
        case 'time':  # time mode
            # deconstruct file name, map from list of str to tuple of int, convert to datetime obj, convert to epoch s
            x = datetime.datetime(*tuple(map(int, file.split('_')[:6]))).timestamp() * x
    return x, f, response
    # collapse all lists of f values into one list that includes them all
    # sweeps and pulses (and even multiple sweeps) have different frequency values.
    # X, f = np.meshgrid()
    # if a data value does not exist for this frequency, set this point to NaN to avoid plotting


def wireplot(ax, x, f, response):
    # plots a line of data
    ax.plot(x, f, zs=response)
    # ax.plot_wireframe(X, f, response, rstride=1, cstride=1)
