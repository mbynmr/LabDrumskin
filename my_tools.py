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
#     # need to look at edge cases: e.g. where (max - min) = 1000 exactly.
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

    # note the time
    end_time = time.localtime()[0:6]
    # load (copy) straight away to avoid conflicts and things
    a = np.loadtxt(f"outputs/output.txt")

    # test details
    if temperature is None:
        temperature = input("Temperature ('rt', '40', etc.):")
    if sample is None:
        sample = input("Sample name ('C0', 'CF4', etc.):")
    if "\n" in sample or "\r" in sample or "\t" in sample or "\b" in sample or "\\" in sample:
        sample = "placeholder_name"
        print("bad sample name, replacing with 'placeholder_name'")
    if save_path is None:
        save_path = input("Write the path to the folder you want to save in (can be 'outputs')")

    # add the test details to the file name
    fname = '_'.join([str(e).zfill(2) for e in end_time]) + f"_{method}_{sample}_{temperature}"

    # save it
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


def time_from_filename(file):
    s = file.split("_T")[0].split("_")
    t = float(s[2]) * 60 * 60 * 24
    t += float(s[3]) * 60 * 60
    t += float(s[4]) * 60
    try:
        t += float(s[5])
    except IndexError:  # old files don't have seconds and throw up this error
        pass
    return t


def temp_from_filename(file):
    try:
        temp_in_file = float(file.split('.txt')[0].split('_')[-1])  # temperature from the file name
    except ValueError:
        return -1

    # 2025_03_10_13_52_51
    # 2025, 03, 10, 13, 52, 51
    # [yyy, mm, dd, hh, mm, ss]
    # set midnight 2020/01/01 as 0 time
    # set a month as always 31 days (who cars). make 'date' be time after 2020 in hours.
    s = file.split("_T")[0].split("_")
    date = (float(s[0]) - 2020) * 12 * 31 * 24
    date += float(s[1]) * 31 * 24
    date += float(s[2]) * 24
    date += float(s[3])

    # if date < (((4 * 12) + 4) * 31 + 25) * 24:  # before 25/04/2024 is temp_get_old. files are claimed to be updated
    #     temp_corr = 1
    if date < (((5 * 12) + 3) * 31 + 6) * 24:  # before 06/03/2025 is temp_get_not_new
        temp_corr = 1  # convert from temp_get_not_new to temp_get
        # return (((100 - 41.294) / (-1.785 - 0)) * np.asarray(voltage)) + 41.294
        temp_corr = temp_get_corr((temp_in_file - 41.294) / ((100 - 41.294) / (-1.785 - 0)))
    elif date < (((5 * 12) + 3) * 31 + 10) * 24 + 14:  # before 10/03/2025 @ 14:00 exactly is temp_get_nearly_new
        temp_corr = 1  # convert from temp_get_nearly_new to temp_get  # todo ?? when is this and what is thehhhuuuu
    elif date < (((5 * 12) + 7) * 31 + 9) * 24 + 9:  # before 09/07/2025 @09:00 is temp_get_corr, correctly labelled.
        temp_corr = temp_in_file
    elif date < (((5 * 12) + 7) * 31 + 16) * 24 + 13:  # before 16/07/2025 @13:00 is mix of old temp_get & new battery
        # voltage has temp_get_corr applied but needed temp_get_nb
        # to undo this, we reverse temp_get_corr. then we apply temp_get_nb.
        undone_voltage = (temp_in_file - 40.3593629746) / (-31.098754907792)
        # (T - c) / m = V
        temp_corr = temp_get_nb(undone_voltage)
    elif date < (((5 * 12) + 9) * 31 + 15) * 24 + 14:  # before 15/09/2025 @14:00 is new battery but drifting? idk. flat
        undone_voltage = (temp_in_file - 56.57646318) / (- 31.46633103)
        temp_corr = temp_get_nb_recal(undone_voltage)
    else:  # after 15/09/2025 @14:00 is the current temp_get
        temp_corr = temp_in_file
    return temp_corr


def temp_get(voltage):  # processes an array of voltages to return the corresponding array of temps (can be len = 1)
    return temp_get_nb_recal(voltage)


def temp_get_nb_recal(voltage):
    return 50.12788 - 31.96931 * np.asarray(voltage)
    # new 0C: 1.38V, new 100C: -1.798V
    # T = mV + c
    # m = -31.46633103
    # c = 56.57646318
    # 15/09/2025 @14:00 this was implemented into the code. files from this time onward are happy


def temp_get_nb(voltage):
    return 56.57646318 - 31.46633103 * np.asarray(voltage)
    # 09/07/2025 @09:00 I put a new battery in the CJC which made all temps appear lower than they used to (+should be).
    # new 0C: 1.568V, new 100C: -1.560V
    # T = mV + c
    # m = 31.969309462915601023017902813299
    # c = 50.127877237851662404092071611253
    # 16/07/2025 @13:00 this was implemented into the code. files from this time onward are happy


def temp_get_corr(voltage):
    return 40.3593629746 - 31.098754907792 * np.asarray(voltage)
    # comes from fit to what the temperature controller said at a series of temperatures
    # the temperature controller being accurate we can't trust 100%. in boiling water it got 100.5 consistently
    # 10/03/2025 @ 14:00 exactly first time this function replaced 'temp_get_nearly_new' in data
    # files ARE NOT UPDATED. KEEP FILES THE SAME if they are created before this point.


def temp_get_nearly_new(voltage):
    return (((100 - 41.31) / (-1.893 - 0)) * np.asarray(voltage)) + 41.31
    # these values come from boiling at -1.893V and unplugged readingggggggggg?????????? don't be stupid that's so wrong
    # this is stupid and wrong: why tf go off 41.31 for the zero? maybe look up what it should be lol.
    # 06/03/2025 first time this function replaced 'temp_get_not_new' in data
    # change 100C from xx to xx
    # files ARE NOT UPDATED. KEEP FILES THE SAME


def temp_get_not_new(voltage):
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


def temp_to_str(temp):
    # converts any float to a string with 2 decimal places
    tempstr = f"{temp:.2f}"  # 2 decimal places
    if len(tempstr.split('.')) < 2:  # if both trailing 0s have been removed
        return tempstr + '.00'
    elif len(tempstr.split('.')[-1]) == 1:  # if one trailing 0 has been removed
        return tempstr + '0'
    return tempstr
    # # works for any temp from 10.00 to 99.99, not sure about any others though!
    # # I might even just want 2dp all the time, rather than 4 sig figs
    # tempstr = f"{temp:.4g}"
    # if len(tempstr.split(".")) == 1:
    #     if len(tempstr) < 4:
    #         return tempstr + ".00"
    #     else:
    #         return tempstr + ".0"
    # elif len(tempstr) < 5:
    #     return tempstr + "0"
    # return tempstr


def strip_nums(s: str):  # strips off trailing numbers if any element is a letter
    # for i in range(10):
    #     s = s.split(f'{i}')[0]
    while np.any(e.isalpha() for e in s) and not s[-1].isalpha:
        s = s[:-2]
    return s


def normalise(v):
    # put v in the range -1 to 1. Not very robust, can be broken by 0s or a len of <2
    v = np.array(v)
    v = v - np.amin(v)
    return (2 * v / np.amax(v)) - 1

