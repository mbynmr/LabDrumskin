import numpy as np
import nolds as nd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def get_from_file(num=None):
    Tk().withdraw()  # so keep the root window from appearing
    filename = askopenfilename()

    s = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents\Shared - Mechanical Vibrations of " \
        r"Ultrathin Films\Lab\data\PSY\Shapes" \
        r"\Semicircle\raw" \
        r"\PSY25sc_5_S10.0__6469.0_19_44.txt"
    # arr = np.loadtxt(s)[:num, :]
    if num is None:
        arr = np.loadtxt(filename)[:, :]
    else:
        arr = np.loadtxt(filename)[:num, :]
    t = arr[:, 0]
    x = arr[:, 1]
    F = arr[:, 2]

    return t, F, x


def embed_ts(x, m, tau):
    N = len(x) - (m - 1) * tau
    return np.column_stack([x[i:i + N] for i in range(0, m * tau, tau)])


def reconstruct_phasespace_trajectory(x, tau, m):
    return embed_ts(x, m, tau)


def autocorr(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    acf = result[result.size // 2:]
    return acf / acf[0]


def estimate_delay(ac):
    # first zero crossing
    zeros = np.where(np.diff(np.sign(ac)))[0]
    if len(zeros) > 0:
        return zeros[0]
    # fallback: first local minimum
    for i in range(1, len(ac)-1):
        if ac[i] < ac[i-1] and ac[i] < ac[i+1]:
            return i
    return 1
    # ac = nd.auto_corr(x)
    # # first zero crossing
    # idx = np.where(np.diff(np.sign(ac)))[0]
    # return int(idx[0]) if len(idx) > 0 else 1


def plotting_poincare():
    # yoink from file
    t, F, x = get_from_file(int(3e4))  # int(1e4)

    ac = autocorr(x)
    tau = estimate_delay(ac)
    # tau = nd.estimate_delay(x)
    m = 10  # embedding dimension
    # smooth low-dimensional systems (Lorenz, Rössler): m = 5–10
    # biological sensors / EEG / ECG: m = 10–20
    # highly noisy or complex signals: m = 15–30
    # mechanical vibration / displacement time series: m = 8–12 (common range)

    X = reconstruct_phasespace_trajectory(x, tau, m)

    lle = nd.lyap_r(x, emb_dim=m, tau=tau, min_tsep=10)
    print(lle)
    # LLE	        Meaning
    # > 0.05	    Strong chaos
    # 0.005 – 0.05	Weak chaos / noise
    # ≈ 0	        Periodic or quasiperiodic
    # < 0	        Convergent or stable
    # PSY25sc_5_S10.0__6469.0_19_44 isssss 0.0040970970813493345
    # 7470 is 0.0017555066965576577

    cd = nd.corr_dim(x, emb_dim=m)
    print(cd)
    # Correlation dimension	    System type
    # ~1	                    limit cycle
    # ~2	                    torus
    # 2–3.5                     fractional	chaotic attractor
    # >>5	                    noise dominates
    # PSY25sc_5_S10.0__6469.0_19_44 isssss 3.01956313919753
    # 7470 is 2.8598725534759226

    fd = 6469.0  # driving frequency  # todo
    dt = 1 / 250000  # sampling rate
    if t[1] != dt:
        raise ValueError  # quits if the first timestep is not the same as the sample rate todo
    T = int(1 / (fd * dt))

    poincare = X[::T]

    plt.scatter(poincare[:, 0], poincare[:, 1], s=3)
    plt.xlabel('x(t)')
    plt.ylabel('x(t + τ)')
    # plt.show()
