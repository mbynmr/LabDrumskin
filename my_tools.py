import numpy as np


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
    try:
        while True:
            with open(f"outputs/{sample_name}_{test_num}.txt") as a_file:
                j += 1
            if j > (10 ** filler) - 1:
                with open(f"outputs/output.txt", 'r') as out:  # reopen the just saved output file in read mode
                    with open(f"outputs/output_error.txt", 'w') as new_out:
                        for line in out:  # faster way of doing this? It won't take long either way
                            new_out.write(line)
                raise FileExistsError(f"Somehow {j + 1} unique files with sample name '{sample_name}' exist.\n"
                                      f"The data is saved in 'output.txt' and now also in 'output_error.'")
            test_num = str(j).zfill(filler)
    except FileNotFoundError:
        print(f"Copying to file name '{sample_name}_{test_num}.txt'")
    np.savetxt(f"outputs/{sample_name}_{test_num}.txt", np.loadtxt(f"outputs/output.txt"), fmt='%.6g')
    # the below method is (probably) slower than savetxt of loadtxt
    # with open(f"outputs/output.txt", 'r') as out:  # reopen the just saved output file in read mode
    #     with open(f"outputs/{sample_name}_{test_num}.txt", 'w') as new_out:
    #         for line in out:
    #             new_out.write(line)
