import numpy as np


def round_sig_figs(x, p):  # credit for this significant figures function https://stackoverflow.com/revisions/59888924/2
    mags = 10 ** (p - 1 - np.floor(np.log10(np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1)))))
    return np.round(x * mags) / mags
