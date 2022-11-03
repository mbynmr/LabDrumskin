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


def resave_output(method="", freqstep="", t=""):
    # saves output.txt under another name
    print("Output saved in 'output.txt'. Enter sample and test details to save a copy with a unique name...")
    temperature = input(f"Temperature ('rt', '40', etc.):")
    sample_name = input(f"Sample name ('C0', 'CF4', etc.):")

    if len(sample_name.split("_")) == 1:  # if no underscores
        # # inside the if it is expected only a sample name and method (S-C1) are added.
        name = method + '-' + temperature + '-' + sample_name  # add some test details to the file name
        j = 0
        filler = 2  # length of zero padding
        test_num = str(j).zfill(filler)
        while True:
            fname = f"{name}_{test_num}.txt"
            try:
                # print(f"'{fname.split('.txt')[0].split(',')[0] + '.txt'}'")
                with open(f"outputs/{fname}") as file:
                    # print(f"'{fname.split('.txt')[0].split(',')[0] + '.txt'}' exists in outputs")
                    j += 1
                if j > (10 ** filler) - 1:
                    np.savetxt(f"outputs/output_overwrite_error.txt", np.loadtxt("outputs/output.txt"), fmt='%.6g')
                    raise FileExistsError(f"Somehow {j + 1} unique files with name '{name}' exist.\n"
                                          f"The unsorted data is saved in 'output.txt' & 'output_overwrite_error.txt'")
                test_num = str(j).zfill(filler)
            except FileNotFoundError:
                break
    else:
        fname = f"{sample_name}aa.txt"

    print(f"Copying to file name '{fname}' and sorting by frequency")
    a = np.loadtxt(f"outputs/output.txt")
    np.savetxt(f"outputs/{fname}", a[np.argsort(a, axis=0)[:, 0]], fmt='%.6g')
    # this also sorts by frequency (or whatever is in column 0)!


def toggle_plot(fig):
    # credit: https://stackoverflow.com/a/32576093
    fig.set_visible(not fig.get_visible())
    plt.draw()
