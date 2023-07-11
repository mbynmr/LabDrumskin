import numpy as np
import os
from tqdm import tqdm

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
        a[i, 3] = file.split("_")[-1].split(".")[0]

    a = a[np.nonzero(a[:, 1])]  # ignore non-related files
    a[:, 0] = a[:, 0] - a[a[:, 0].argmin(), 0]
    # todo could sort

    np.savetxt("outputs/outtimes.txt", a, fmt='%.6g')
    if name is not None:
        np.savetxt(save_path + "/" + name, a, fmt='%.6g')
        print(f"saved as file name '{name}'")
