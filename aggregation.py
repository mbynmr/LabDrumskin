import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from my_tools import resave_auto


def manual_peak(save_path, cutoff):
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
    resave_auto(save_path=save_path, manual=True)


def on_pick(event):
    with open("outputs/manual.txt", 'w') as m:
        m.writelines(f"{event.artist.get_xdata()[event.ind[0]]} {-2} {-2}")


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
            # todo atm it just searches for nearest temperature. I want nearest temp with lowest error data[:, 2]
            ns = np.argwhere((np.abs(data[:, 0] - t)) < 1)  # if less than 1C out
            if len(ns) != 0:
                data_at_temps[i] = data[ns[data[ns, 2].argmin()], 1]
        film = file_name.split('.')[0].split('_')[-1].split(' ')[0]
        film_list.append(film)
        dic[film] = data_at_temps

    file_path = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" +\
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
            # todo atm it just searches for nearest temperature. I want nearest temp with lowest error data[:, 2]
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


def strip_nums(s: str):  # strips off trailing numbers if any element is a letter
    # for i in range(10):
    #     s = s.split(f'{i}')[0]
    while np.any(e.isalpha() for e in s) and not s[-1].isalpha:
        s = s[:-2]
    return s
