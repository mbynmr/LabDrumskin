import numpy as np
import matplotlib.pyplot as plt


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


def resave_output():
    # saves output.txt under another name
    sample_name = input(f"Output saved in 'output.txt'. If you don't want it to be overwritten, enter sample name:")
    j = 0
    filler = 3  # length of zero padding
    test_num = str(j).zfill(filler)
    while True:
        fname = f"{sample_name}_{test_num}.txt"
        try:
            with open(f"outputs/{fname}") as a_file:
                j += 1
            if j > (10 ** filler) - 1:
                np.savetxt(f"outputs/output_overwrite_error.txt", np.loadtxt("outputs/output.txt"), fmt='%.6g')
                raise FileExistsError(f"Somehow {j + 1} unique files with sample name '{sample_name}' exist.\n"
                                      f"The unsorted data is saved in 'output.txt' and in 'output_overwrite_error.txt'")
            test_num = str(j).zfill(filler)
        except FileNotFoundError:
            print(f"Copying to file name '{fname}' and sorting by frequency")
            break
    a = np.loadtxt(f"outputs/output.txt")
    np.savetxt(f"outputs/{fname}", a[np.argsort(a, axis=0)[:, 0]], fmt='%.6g')
    # this also sorts by frequency (or whatever is in column 0)!


def toggle_plot(fig):
    # credit: https://stackoverflow.com/a/32576093
    fig.set_visible(not fig.get_visible())
    plt.draw()
