import numpy as np
import matplotlib.pyplot as plt
import time
from subprocess import check_call


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


# def ax_lims_wip(data):
#     # return cand be fed directly into ax.set_ylim() or ax.set_xlim()
#     if len(data) < 2:
#         return [min(data) - 1, max(data) + 1]
#
#     # todo edge cases: e.g. where (max - min) = 1000 exactly.
#     mag = int(0)
#     if np.isclose(max(data), min(data), rtol=0, atol=10 ** mag):
#         up = False
#     else:
#         up = True
#
#     while bool(np.isclose(max(data), min(data), rtol=0, atol=10 ** mag)) ^ up:
#         if up:
#             mag += 1
#         else:
#             mag -= 1
#
#     # mag so far is the magnitude of the first difference between the numbers.
#     # sig is how many sig figs we need to make mag visible in the number.
#     # [101, 105] (diff 4) mag is 1, sig is 3+1
#     # [101, 205] (diff 204) mag is 3, sig is 1+1
#
#     if up:
#         sig = mag - 1
#     sig = abs(mag)
#     return (round_sig_figs(min(data), sig, 'f'),
#             round_sig_figs(max(data), sig, 'c'))


def ax_lims(data):
    # make axis limits with clearance on both sides even after being rounded to 2 significant figures
    # 5% of the range between the max and min is the minimum clearance
    # return cand be fed directly into ax.set_ylim() or ax.set_xlim()
    diff = (max(data) - min(data)) * 2.5 / 100
    return (round_sig_figs(min(data) - diff, 4, 'f'),
            round_sig_figs(max(data) + diff, 4, 'c'))


def resave_output(method=None, save_path=None, temperature=None, sample=None, copy=False):
    # saves output.txt under another name

    end_time = time.localtime()[0:6]

    # load (copy) straight away to avoid conflicts and things
    a = np.loadtxt(f"outputs/output.txt")

    if temperature is None:
        temperature = input("Temperature ('rt', '40', etc.):")
        # temperature = 20
    if sample is None:
        sample = input("Sample name ('C0', 'CF4', etc.):")
        # sample_name = "PSY2_J_" + f"{np.random.random():.6g}".split(".")[1]
    elif "\n" in sample or "\r" in sample or "\t" in sample or "\b" in sample or "\\" in sample:
        sample = "placeholder_name"
        print("bad sample name, replacing with 'placeholder_name'")
    if save_path is None:
        save_path = input("Write the path to the folder you want to save in (can be 'outputs')")

    # add some test details to the file name
    fname = '_'.join([str(e).zfill(2) for e in end_time]) + f"_{method}_{sample}_{temperature}"

    # a[np.argsort(a, axis=0)[:, 0]] sorts by frequency (or whatever is in column 0)!

    print(f"Copying to file name '{fname}.txt' and sorting by frequency", end='')
    # np.savetxt(f"{save_path}/{fname}.txt", a[np.argsort(a, axis=0)[:, 0]], fmt='%.4g')
    np.savetxt(f"{save_path}/{fname}.txt", a[np.argsort(a, axis=0)[:, 0]], fmt='%.4g')
    if copy:
        print("\n", end='')
        copy2clip(fname)
    # while True:
    #     try:
    #         np.loadtxt(f"{save_path}/{fname}.txt")
    #     except FileNotFoundError:
    #         print(f"\rCopying to file name '{fname}.txt' and sorting by frequency", end='')
    #         np.savetxt(f"{save_path}/{fname}.txt", a[np.argsort(a, axis=0)[:, 0]], fmt='%.4g')
    #         return f"{save_path}/{fname}.txt"
    #     fname = fname + "_"


def resave_auto(save_path="outputs", sample_name=None, method=None):
    # saves output.txt under another name

    end_time = time.localtime()[0:5]

    # load (copy) straight away to avoid conflicts and things
    m = np.loadtxt(f"outputs/manual.txt")

    if sample_name is None:
        sample_name = input("Sample name ('C0', 'CF4', etc.):")
    if method is None:
        method = input("Method (S, A ,P):")
    if save_path == "outputs":
        save_path = input("Write the path to the folder you want to save in (can be 'outputs' or blank)")

    # add some test details to the file name
    fname = '_'.join([str(e).zfill(2) for e in end_time]) + f"_T{method}m_{sample_name}.txt"
    print(f"Copying 'manual.txt' to file name '{fname}'")
    np.savetxt(f"{save_path}/{fname}", m, fmt='%.6g')


def toggle_plot(fig):
    # credit: https://stackoverflow.com/a/32576093
    fig.set_visible(not fig.get_visible())
    plt.draw()


def copy2clip(txt):
    # credit: https://stackoverflow.com/a/41029935
    print(f"Clipboard is {txt}")
    return check_call('echo ' + txt.strip() + '|clip', shell=True)


def temp_get(voltage):  # processes an array of voltages to return the corresponding array of temps (can be len = 1)
    return (((100 - 41.294) / (-1.785 - 0)) * np.asarray(voltage)) + 41.294
    # 25/04/2024
    # change 100C from -1.600 to -1.785
    # change 0 point from measurement of 0C with ice to reported temp when 0V (disconnected) of 41.294C
    # all important files have been updated but there is a list of data files that haven't, that needs to be automated.


def temp_get_old(voltage):
    return (100 / (-1.600 - 1.255)) * (np.asarray(voltage) - 1.255)


def temp_get_oldest(voltage):
    zero = 0.055  # is probably 1 degree, and is from 0.051 to 0.055 pretty much
    hundred = -3.44
    # fifty = -1.62  # approximately
    # eighty_five = -2.88  # approximately
    # ninety_six = -3.24
    # temp = (100 / (hundred - zero)) * voltage - zero
    return (100 / (hundred - zero)) * (np.asarray(voltage) - zero)


def convert_temp_to_tempstr(temp):
    # works for any temp from 10.00 to 99.99, not sure about any others though!
    tempstr = f"{temp:.4g}"
    if len(tempstr.split(".")) == 1:
        return tempstr + ".00"
    elif len(tempstr) < 5:
        return tempstr + "0"
    return tempstr
