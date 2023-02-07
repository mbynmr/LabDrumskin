import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


def aggregate(a_or_s):
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
                '2': empty,
                '3': empty,
                '4': empty,
                '4.5': empty,
                '5.5': empty,
                '6': empty,
                '7': empty,
                '9': empty,
                }  # [2, 3, 4, 4.5, 5.5, 6, 7, 9]    # [1/2, 1/3, 1/4, 1/4.5, 1/5.5, 1/6, 1/7, 1/9]
            # radii: [1, 1.5, 2, 2.25, 2.75, 3, 3.5, 4.5]    # [1/1, 1/1.5, 1/2, 1/2.25, 1/2.75, 1/3, 1/3.5, 1/4.5]
        case _:
            raise ValueError

    # folder_name = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" +\
    #               r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\Temperature Sweeps"
    for key in dic.keys():
        if key == 'temps':
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
            ns = np.argwhere((np.abs(data[:, 0] - t)) < 1)
            if len(ns) != 0:
                n = ns[data[ns, 2].argmin()]
                if (np.abs(data[n, 0] - t)) < 1:  # if less than 1C out
                    data_at_temps[i] = data[n, 1]
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
