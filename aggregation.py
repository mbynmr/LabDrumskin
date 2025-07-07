import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from tqdm import tqdm
import sys

from my_tools import resave_auto, time_from_filename, temp_from_filename
from fitting import fit_agg, fit


def resave(save_path, name=None):
    spectra_path = save_path + r"\Spectra"
    # save_path = "C:/Users/mbynmr/OneDrive - The University of Nottingham/Documents/Shared - Mechanical Vibrations of Ultrathin Films/Lab/data/PIM/6 percent/auto"
    # print(os.listdir(save_path))
    files = os.listdir(spectra_path)
    a = np.zeros([len(files), 4])

    for i, file in tqdm(enumerate(files), total=len(files)):
        # if file.split("_")[-1] != "T.txt":
        if file.split(".")[-1] != "txt":
            continue

        cutoff = [0.105, 0.3]
        # method = "fit then max"
        method = "max then fit"  # maybe the best?
        # method = "max"
        # method = "fit"

        match method:
            case "fit then max":
                peak, error = fit(spectra_path + "/" + file, cutoff=cutoff, copy=False)
                data = np.loadtxt(spectra_path + "/" + file)
                data = data[np.argsort(data, axis=0)[:, 0]]  # sort it!
                cutoff = [(peak[1] - 50) / 1e4, (peak[1] + 50) / 1e4]
                data = data[int(len(data[:, 0]) * cutoff[0]):int(len(data[:, 0]) * cutoff[1])]
                peak = data[np.argmax(data[:, 1]), 0]
                error = 2.5
            case "max then fit":
                data = np.loadtxt(spectra_path + "/" + file)
                data = data[np.argsort(data, axis=0)[:, 0]]  # sort it!
                data = data[int(len(data[:, 0]) * cutoff[0]):int(len(data[:, 0]) * cutoff[1])]
                peak = data[np.argmax(data[:, 1]), 0]
                peak, error = fit(spectra_path + "/" + file, cutoff=[(peak - 50) / 1e4, (peak + 50) / 1e4], copy=False)
            case "max":
                data = np.loadtxt(spectra_path + "/" + file)
                data = data[np.argsort(data, axis=0)[:, 0]]  # sort it!
                data = data[int(len(data[:, 0]) * cutoff[0]):int(len(data[:, 0]) * cutoff[1])]
                peak = data[np.argmax(data[:, 1]), 0]
                error = 2.5
            case "fit":
                peak, error = fit(spectra_path + "/" + file, cutoff=cutoff, copy=False)

        s = file.split("_TP")[0].split("_")
        # 2023_05_19_11_18
        # t = float(s[0]) * 60 * 60 * 24 * 365
        t = float(s[2]) * 60 * 60 * 24
        t += float(s[3]) * 60 * 60
        t += float(s[4]) * 60
        try:
            t += float(s[5])
        except IndexError:  # old files don't have seconds and throw up this error
            continue  # todo
            pass
        if type(error) is not np.float_:
            continue
        a[i, 0] = t
        a[i, 1] = peak
        a[i, 2] = error
        temp = file.split("_")[-1].split(".")[0]
        if isinstance(temp, str):
            a[i, 3] = 0
        else:
            a[i, 3] = temp

    a = a[np.nonzero(a[:, 1])]  # ignore non-related files
    a[:, 0] = a[:, 0] - a[a[:, 0].argmin(), 0]
    # todo could sort

    np.savetxt("outputs/outtimes.txt", a, fmt='%.6g')
    if name is not None:
        np.savetxt(save_path + "/" + name, a, fmt='%.6g')
        print(f"saved as file name '{name}'")


def manual_peak(save_path, cutoff, file_name=None):
    start = 70
    stop = 40
    step = 2.5
    temps_all = np.linspace(start=start, stop=stop, num=int(1 + abs(start - stop) / step))
    temps_should_be = np.repeat(np.linspace(start=start, stop=stop, num=int(1 + abs(stop - start) / 2.5)), repeats=3)
    temps = np.zeros_like(temps_all)
    peaks = np.zeros_like(temps)

    x = [0, 5]  # todo find another way of stopping pycharm from being angry with me for 'data[:, 2] = abs(x[0] - x[1])'
    for i, t in enumerate(temps_all):
        t = temps_all[i]
        root = tk.Tk()
        root.withdraw()
        print(f"\rit {str(len(temps_should_be) - 1 - (3 * i + 1)).zfill(len(str(len(temps_all))))}, temp should be {t}",
              end='')
        if file_name is None:
            file_name = filedialog.askopenfilename()
        # sample = file_name.split('.txt')[-1].split('_')[-2]
        temp = file_name.split('/')[-1].split('.txt')[0].split('_')[-1]
        temps[i] = temp
        # print(f"temp turned out to be {temp}")

        data = np.loadtxt(file_name)
        data = data[int(len(data[:, 0]) * cutoff[0]):int(len(data[:, 0]) * cutoff[1])]
        x = data[:, 0]
        y = data[:, 1]
        fig, ax = plt.subplots()  # figsize=(16, 9)
        plt.get_current_fig_manager().full_screen_toggle()
        ax.plot(x, y, '.', picker=10)
        fig.canvas.callbacks.connect('pick_event', on_pick)
        plt.draw()
        with open("outputs/manual.txt", 'w') as m:
            m.writelines(f"{-1} {-1}")
        while np.loadtxt("outputs/manual.txt")[-1] == -1:
            plt.waitforbuttonpress()
        peaks[i] = np.loadtxt("outputs/manual.txt")[0]

        plt.close(fig)

    data = np.zeros([len(temps), 3])
    data[:, 0] = temps
    data[:, 1] = peaks
    data[:, 2] = abs(x[0] - x[1])
    np.savetxt("outputs/manual.txt", data, fmt='%.6g')
    print("\ndone!")
    resave_auto(save_path=save_path)


def manual_peak_auto(save_path, cutoff, method, sample=None, printer=None):
    if sample is None:
        sample = input("enter sample name you are looking at plz:")
    # if cutoff is None:
    #     cutoff = [0, 1]
    if printer is None:
        printer = sys.stderr
    if "/AutoTemp" not in save_path and r"\AutoTemp" not in save_path:
        save_path = save_path + "/AutoTemp"
        if "/Spectra" not in save_path and r"\Spectra" not in save_path:
            spectra_path = save_path + "/Spectra"
        else:
            spectra_path = save_path
    else:
        spectra_path = save_path

    files = os.listdir(spectra_path)
    files.sort()
    indexes = []
    for i in range(len(files)):
        f = files[i]
        try:
            if f[4] != '_' or f[7] != '_' or f[10] != '_' or f[13] != '_' or f[16] != '_':
                print(f'removing {f} at {i}')
                indexes.append(i)
        except IndexError:
            print(f'removing {f} at {i}')
            indexes.append(i)
    data = np.zeros([len(files), 4])
    for i in sorted(indexes, reverse=True):
        files.pop(i)
    # for Veusz (not zero indexed D: oh noes)
    # datacol1 = time
    # datacol2 = frequency peak value
    # datacol3 = error in frequency
    # datacol4 = temperature
    # print(files)
    print('press mouse on datapoint to select, or n to skip to next spectra. press x to quit.')

    prevfile = None
    skip = 0
    maxy = 5
    for i, file in tqdm(enumerate(files), total=len(files), file=printer, ncols=42):
        # sample name from the file name
        # 2023_06_22_12_42_26_TP211_PSY2_4_81.12.txt
        # sample_name = "_".join(file.split('.txt')[0].split('_T')[-1].split('_')[1:-1])
        # 2024_05_22_12_30_42_TS01_PSY125_12_80.34.txt
        sample_name = '_'.join(file.split('.txt')[0].split('_T')[-1].split('_')[1:3])
        if sample_name != sample:
            skip += 1
            continue

        if method == 'B' and file.split('_')[6][0:2] == 'TP':
            prevfile = file
            skip += 1
            continue

        xy = np.loadtxt(spectra_path + "/" + file)
        # xy = xy[int(len(xy[:, 0]) * cutoff[0]):int(len(xy[:, 0]) * cutoff[1]), ...]
        fig, ax = plt.subplots()  # figsize=(16, 9)
        plt.get_current_fig_manager().full_screen_toggle()
        ax.plot(xy[:, 0], xy[:, 1], '-D', c='blue', mfc='red', mec='k', picker=10)
        ax.set_xlim(cutoff)
        continuing = True
        while continuing:
            actual = np.amax(np.where(xy[:, 0] <= cutoff[1], xy[:, 1], 0))
            argg = np.argwhere(xy[:, 1] == actual)
            if argg == 0:
                continuing = False
            try:
                if xy[argg, 0] > 3 * xy[argg + 1, 1] or xy[argg, 1] > 3 * xy[argg - 1, 1]:
                    xy[argg, 0] = 0
                else:
                    continuing = False
            except IndexError:  # cba fixing this more than just a try except lol
                continuing = False
        while maxy > 1.4 * actual:  # it only covers 60% of the screen, make that more. otherwise gdgd!
            maxy = 0.95 * maxy
        while maxy < 1.1 * actual:
            maxy = 1.01 * maxy
        ax.set_ylim([0, maxy])
        if method == 'B':  # P & S both 'B' method: plot the line of both (first pulse then sweep).
            xyp = np.loadtxt(spectra_path + "/" + prevfile)
            xyp[:, 1] = xyp[:, 1] * (0.95 * actual / np.amax(xyp[:, 1]))
            ax.plot(xyp[:, 0], xyp[:, 1], '-b')
        # markeredgecolor       mec     color
        # markeredgewidth       mew     float
        # markerfacecolor       mfc     color
        # markerfacecoloralt    mfcalt  color
        ax.set_title(f"picker {i - skip} of {len(files) - skip} (roughly)")
        fig.canvas.callbacks.connect('pick_event', on_pick)
        fig.canvas.callbacks.connect('key_press_event', on_skip)
        plt.tight_layout()
        plt.draw()
        with open("outputs/manual.txt", 'w') as m:
            m.writelines(f"{-1} {-1}")
        while np.loadtxt("outputs/manual.txt")[-1] == -1:
            butt = plt.waitforbuttonpress()  # returns True for key, False for mouse, None if timeout before button press.
            if not butt:  # if mouse click, file should be updated. read the file manual.txt
                data[i, 0] = time_from_filename(file)
                data[i, 1] = np.loadtxt("outputs/manual.txt")[0]  # take the first entry (the useful data)
                data[i, 2] = 10  # todo error in Hz
                data[i, 3] = temp_from_filename(file)
                # print(data[i, ...])
            elif np.loadtxt("outputs/manual.txt")[0] == -3123482:
                print('quitting manual peaks. NOT SAVING data.')  # todo option to save data as well? should be easy.
                plt.close(fig)
                return
            elif np.loadtxt("outputs/manual.txt")[0] == -912384:
                print(f'skipping spectra number {i}')
        plt.close(fig)

    # data = data[np.argwhere(data[:, 2] != 0).flatten(), ...]
    data = data[~np.all(data == 0, axis=1)]
    # data.sort(axis=0)
    data[:, 0] = data[:, 0] - data[0, 0]
    np.savetxt("outputs/manual.txt", data, fmt='%.6g')
    print("\ndone!")
    resave_auto(save_path=save_path, sample_name=sample, method=file.split("_T")[1][0])


def on_pick(event):
    with open("outputs/manual.txt", 'w') as m:
        m.writelines(f"{event.artist.get_xdata()[event.ind[0]]} {-2} {-2}")


def on_skip(event):
    if event.key == 'x':
        with open("outputs/manual.txt", 'w') as m:
            m.writelines(f"{-3123482} {-3} {-3}")
    elif event.key == 'n':
        with open("outputs/manual.txt", 'w') as m:
            m.writelines(f"{-912384} {-4} {-4}")


def aggregate():
    temps = np.linspace(start=15, stop=70, num=int(1 + abs(70 - 15) / 5))  # 15, 20, ..., 70
    empty = np.zeros_like(temps)
    dic = {'temps': temps}
    films = {'C21': 6, 'C22': 6, 'C23': 7, 'C24': 6, 'C25': 5.5, 'C26': 4.5, 'C27': 3, 'C28': 2, 'C29': 6, 'C30': 4,
             'C31': 7, 'C32': 6, 'C33': 4, 'C34': 4, 'C35': 5.5, 'C36': 4.5, 'C37': 3, 'C38': 4, 'C39': 100, 'C40': 2,
             'C41': 2, 'C42': 10, 'C43': 10, 'C44': 4, 'C45': 7}

    # folder_name = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" +\
    #               r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\Temperature Sweeps"
    film_list = []
    while True:
        root = tk.Tk()
        root.withdraw()
        file_name = filedialog.askopenfilename()
        data = np.loadtxt(file_name)
        if data[0, 2] == -1:
            break  # end if selected end!
        data_at_temps = np.zeros_like(temps)
        for i, t in enumerate(temps):
            # old to do atm it just searches for nearest temperature. I want nearest temp with lowest error data[:, 2]
            ns = np.argwhere((np.abs(data[:, 0] - t)) < 1)  # if less than 1C out
            if len(ns) != 0:
                data_at_temps[i] = data[ns[data[ns, 2].argmin()], 1]
        film = file_name.split('.')[0].split('_')[-1].split(' ')[0]
        film_list.append(film)
        dic[film] = data_at_temps

    file_path = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" + \
                r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\PDMS\AutoTemp\chosen"
    # save dict to file in nice format
    all = []
    data = np.zeros([len(dic.keys()) - 1, len(temps)])

    with open(f"{file_path}/samples.txt", 'w') as cs, open(f"{file_path}/ds.txt", 'w') as ds:
        for i, key in enumerate(dic.keys()):
            print(f"{key}: {dic[key]}")
            if key != 'temps':
                all.append(films[film_list[i - 1]])
                data[i - 1, :] = dic[key]
                cs.writelines(f"{key}\n")
                ds.writelines(f"{films[key]}\n")
    # np.savetxt("outputs/data.txt", data, fmt='%.6g')
    np.savetxt(f"{file_path}/data.txt", data, fmt='%.6g')
    print(all)


def aggregate_old(a_or_s):
    temps = np.linspace(start=15, stop=70, num=int(1 + abs(70 - 15) / 5))  # 15, 20, ..., 70
    empty = np.zeros_like(temps)
    match a_or_s:
        case 's':
            dic = {
                'temps': temps,
                'E': empty,  # 0.200
                'A': empty,  # 0.245
                '1P': empty,  # 0.300
                'EA': empty,  # 0.424
                'C': empty,  # 0.469
                'T': empty,  # 0.557
                'CF': empty  # 0.624
            }  # [0.200, 0.245, 0.300, 0.424, 0.469, 0.557, 0.624]
            # solvent_lookup = {''}
        case 'a':
            dic = {
                'temps': temps,
                '3': empty,  # '2': empty,
                '4': empty,
                '4.5': empty,
                '5.5': empty,
                '6': empty,
                '7': empty,
            }  # [3, 4, 4.5, 5.5, 6, 7]    # [1/3, 1/4, 1/4.5, 1/5.5, 1/6, 1/7]
            # radii in m: [1.5e-3, 2e-3, 2.25e-3, 2.75e-3, 3e-3, 3.5e-3]
            # 1/radius in 1/m: [(1e3)/1.5, (1e3)/2, (1e3)/2.25, (1e3)/2.75, (1e3)/3, (1e3)/3.5]
        case _:
            raise ValueError

    # folder_name = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" +\
    #               r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\Temperature Sweeps"
    for key in dic.keys():
        if key == 'temps':  # or key != '5.5'
            continue
        root = tk.Tk()
        root.withdraw()
        print(f"Use the file dialog to find data for key: {key}")
        file_name = filedialog.askopenfilename()
        sample = file_name.split('.')[-2].split('_')[-1]
        # solvent = strip_nums(sample)  # remove numbers from sample name to find solvent
        data = np.loadtxt(file_name)
        data_at_temps = np.zeros_like(temps)
        for i, t in enumerate(temps):
            # atm it just searches for nearest temperature. I want nearest temp with lowest error data[:, 2]
            ns = np.argwhere((np.abs(data[:, 0] - t)) < 1)  # if less than 1C out
            if len(ns) != 0:
                data_at_temps[i] = data[ns[data[ns, 2].argmin()], 1]
        dic[key] = data_at_temps

    # save dict to file in nice format
    data = np.zeros([len(dic.keys()) - 1, len(temps)])
    for i, key in enumerate(dic.keys()):
        print(f"{key}: {dic[key]}")
        if key != 'temps':
            data[i - 1, :] = dic[key]
    np.savetxt("outputs/data.txt", data, fmt='%.6g')


def colourplot():
    file_name = input("file name and path:")
    x, y = np.meshgrid(np.linspace(start=15, stop=70, num=int(1 + abs(70 - 15) / 5)),
                       [0.200, 0.245, 0.300, 0.424, 0.469, 0.557, 0.624])
    data = np.loadtxt(file_name)
    plt.pcolor(x, y, data, cmap=plt.get_cmap("inferno"))
    plt.show()


def scatter3d(save_path=None, printer=None):
    if save_path is None:
        save_path = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents\Shared - Mechanical " + \
                    r"Vibrations of Ultrathin Films\Lab\data\PSY\Vary temperature\AutoTemp"
    if printer is None:
        printer = sys.stderr
    if "/AutoTemp" not in save_path and r"\AutoTemp" not in save_path:
        save_path = save_path + "/AutoTemp"

    files = os.listdir(save_path)
    files.sort()
    thicknesses = {'125': 54, '2': 104, '250': 147, '3': 182, '225': 121, '09': 32, '9': 32, '150': 44, '1': 41}

    my_dpi = 102
    fig = plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(projection='3d')

    skip = 0
    for i, file in tqdm(enumerate(files), total=len(files), file=printer, ncols=42):
        try:
            method = file.split('.txt')[0].split('_')[5]
        except IndexError:
            skip += 1
            continue
        if method != 'TSm':
            skip += 1
            continue
        sample_name = '_'.join(file.split('.txt')[0].split('_T')[-1].split('_')[1:3])
        st = ''.join([c if c.isdigit() else ' ' for c in sample_name])
        conc = str([int(s) for s in st.split() if s.isdigit()][0])
        data = np.loadtxt(save_path + "/" + file)
        ys = data[:, -1]
        zs = data[:, 1]
        xs = np.ones_like(ys) * thicknesses[conc]

        colours = np.arange(len(ys))
        ax.scatter(xs, ys, zs, c=colours, marker='.')
    plt.show()
