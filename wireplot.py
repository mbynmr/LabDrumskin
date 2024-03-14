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
        # ax.set_xlabel('concentration/temperature/thickness/time')
        ax.set_xlabel('Concentration / %wt')
        ax.set_ylabel('Frequency / Hz')
        ax.set_zlabel('ln(Response / a.u.)')

    files = os.listdir(path)
    files.sort()
    # for i, file in tqdm(enumerate(files), total=len(files), file=printer, ncols=52):
    for i, file in enumerate(files):
        # 2024_03_12_13_11_10_S1V_PSY1_6_20.txt
        if file.split('.')[-1] == 'txt' and len(file.split('_')) >= 10:  # this is specific file name format!!
            x, f, response = data_extractor(path + "/" + file, mode)
            wireplot(ax, x, f, np.log(response))

    plt.show()


def data_extractor(file, mode):
    data = np.loadtxt(file)
    f = data[:, 0]
    response = data[:, 1]
    x = np.ones_like(f)
    match mode:
        case 'conc':  # concentration mode
            x = float(file.split('_')[7][3:]) * x
        case 'thick':  # thickness mode
            lookup_table = 1
            thing = float(file.split('_')[7][3:])
            x = lookup_table * x
        case 'temp':  # temperature mode
            x = float(file.split('_')[-1]) * x
        case 'time':  # time mode
            times = file.split('_')[:6]
            t = int(times[-1])
            t += int(times[-2]) * 60
            t += int(times[-3]) * 60 * 60
            t += int(times[-4]) * 60 * 60 * 24
            # t += int(times[-4]) * 60 * 60 * 24 * 28
            x = t * x
    return x, f, response
    # collapse all lists of f values into one list that includes them all
    # sweeps and pulses (and even multiple sweeps) have different frequency values.
    # X, f = np.meshgrid()
    # if a data value does not exist for this frequency, set this point to NaN to avoid plotting


def wireplot(ax, x, f, response):
    # plots a line of data
    ax.plot(x, f, zs=response)
    # ax.plot_wireframe(X, f, response, rstride=1, cstride=1)
