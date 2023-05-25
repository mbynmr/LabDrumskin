import numpy as np
import os

from fitting import fit


def resave(save_path, name=None):
    spectra_path = save_path + r"\Spectra"
    # save_path = "C:/Users/mbynmr/OneDrive - The University of Nottingham/Documents/Shared - Mechanical Vibrations of Ultrathin Films/Lab/data/PIM/6 percent/auto"
    # print(os.listdir(save_path))
    files = os.listdir(spectra_path)
    a = np.zeros([len(files), 3])
    s = files[0].split("_TP")[0].split("_")
    # t = float(s[0]) * 60 * 60 * 24 * 365
    # t += float(s[2]) * 60 * 60 * 24
    t = float(s[3]) * 60 * 60
    t += float(s[4]) * 60
    t0 = t
    for i, file in enumerate(files):
        if file.split("_")[-1] != "T.txt":
            continue
        peak, error = fit(spectra_path + "/" + file, cutoff=[0.14, 0.24], copy=False)
        s = file.split("_TP")[0].split("_")
        # 2023_05_19_11_18
        # t = float(s[0]) * 60 * 60 * 24 * 365
        # t += float(s[2]) * 60 * 60 * 24
        t = float(s[3]) * 60 * 60
        t += float(s[4]) * 60
        a[i, 0] = t - t0
        a[i, 1] = peak
        a[i, 2] = error

    a = a[np.nonzero(a[:, 1])]  # ignore non-related files
    a[:, 0] = a[:, 0] - a[0, 0]

    np.savetxt("outputs/outtimes.txt", a, fmt='%.6g')
    if name is not None:
        np.savetxt(save_path + "/" + name, a, fmt='%.6g')
        print(f"saved as file name '{name}'")
