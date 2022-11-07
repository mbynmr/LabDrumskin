import numpy as np
import matplotlib.pyplot as plt
import time


def round_sig_figs(x, sig_fig, method='d'):
    # credit for this significant figures function https://stackoverflow.com/revisions/59888924/2
    mags = 10 ** (sig_fig - 1 - np.floor(np.log10(np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (sig_fig - 1)))))
    match method:
        case 'd':  # default - correct rounding
            return np.round(x * mags) / mags
        case 'f':  # floor
            return np.floor(x * mags) / mags
        case 'c':  # ceil
            return np.ceil(x * mags) / mags


def ax_lims(data):
    # make axis limits with clearance on both sides even after being rounded to 2 significant figures
    # 5% of the range between the max and min is the minimum clearance
    # return cand be fed directly into ax.set_ylim() or ax.set_xlim()
    diff = (max(data) - min(data)) * 5 / 100
    return (round_sig_figs(min(data) - diff, 2, 'f'),
            round_sig_figs(max(data) + diff, 2, 'c'))


def resave_output(method=None, save_path="outputs", freqstep=None, t=None):
    # saves output.txt under another name
    print("Output saved in 'output.txt'. Enter sample and test details to save a copy with a unique name...")
    temperature = input("Temperature ('rt', '40', etc.):")
    sample_name = input("Sample name ('C0', 'CF4', etc.):")
    if save_path == "outputs":
        save_path = input("Write the path to the folder you want to save in (otherwise will go to 'outputs')")

    # add some test details to the file name
    name = '_'.join([str(e).zfill(2) for e in time.localtime()[0:5]]) + f"_{method}_{sample_name}_{temperature}"
    j = 0
    filler = 2  # length of zero padding
    while True:
        fname = f"{name}_{str(j).zfill(filler)}.txt"
        try:
            # print(f"'{fname.split('.txt')[0].split(',')[0] + '.txt'}'")
            with open(f"{save_path}/{fname}") as file:
                # print(f"'{fname.split('.txt')[0].split(',')[0] + '.txt'}' exists in outputs")
                j += 1
            if j > (10 ** filler) - 1:
                np.savetxt(f"{save_path}/output_overwrite_error.txt", np.loadtxt("outputs/output.txt"), fmt='%.6g')
                raise FileExistsError(f"Somehow {j + 1} unique files with name '{name}' exist.\n"
                                      f"The unsorted data is saved in 'output.txt' and 'output_overwrite_error.txt'")
        except FileNotFoundError:
            break

    print(f"Copying to file name '{fname}' and sorting by frequency")
    a = np.loadtxt(f"outputs/output.txt")
    # a[np.argsort(a, axis=0)[:, 0]] sorts by frequency (or whatever is in column 0)!
    np.savetxt(f"{save_path}/{fname}", a[np.argsort(a, axis=0)[:, 0]], fmt='%.6g')


def toggle_plot(fig):
    # credit: https://stackoverflow.com/a/32576093
    fig.set_visible(not fig.get_visible())
    plt.draw()
