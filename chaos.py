import numpy as np
import nolds as nd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tqdm import tqdm
import os
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def get_from_file(file=None, num=None):
    if file is None:
        Tk().withdraw()  # so keep the root window from appearing
        filename = askopenfilename()

        s = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents\Shared - Mechanical Vibrations of " \
            r"Ultrathin Films\Lab\data\PSY\Shapes" \
            r"\Semicircle\raw" \
            r"\PSY25sc_5_S10.0__6469.0_19_44.txt"
    else:
        filename = file
    # arr = np.loadtxt(s)[:num, :]
    if num is None:
        arr = np.loadtxt(filename)[:, :]
    else:
        arr = np.loadtxt(filename)[:int(num), :]

    dt = arr[1, 0] - arr[0, 0]
    if dt == 0:
        dt = 1 / 250000
    arr = arr[int(len(arr) / 2):, :]
    t = np.arange(len(arr[:, 0])) * dt
    x = arr[:, 1]
    F = arr[:, 2]
    #'C:\\Users\\mbynmr\\OneDrive - The University of Nottingham\\Documents\\Shared - Mechanical Vibrations of Ultrathin Films\\Lab\\data\\PSY\\Shapes\\Semicircle\\raw\\PSY25sc_5_S10.0__6450.0_0_44.txt'
    # t is 0.1, 0.1, 0.1, 0.1, till 12th which is 0.1001, increments every 20ish

    try:
        filename = filename.split(r'\raw')[1]
    except IndexError:
        pass
    fd = float(filename.split('__')[1].split('_')[0])
    # print(fd)  # 6472
    return t, F, x, fd, dt


def embed_ts(x, m, tau):
    N = len(x) - (m - 1) * tau
    return np.column_stack([x[i:i + N] for i in range(0, m * tau, tau)])


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
    t, F, x, fd, dt = get_from_file(num=int(1e5))  # int(1e4)

    ac = autocorr(x)
    tau = estimate_delay(ac)
    # tau = nd.estimate_delay(x)
    m = 10  # embedding dimension
    # smooth low-dimensional systems (Lorenz, Rössler): m = 5–10
    # biological sensors / EEG / ECG: m = 10–20
    # highly noisy or complex signals: m = 15–30
    # mechanical vibration / displacement time series: m = 8–12 (common range)

    X = embed_ts(x, m, tau)

    # lle = nd.lyap_r(x, emb_dim=m, tau=tau, min_tsep=10)
    # print(lle)
    # LLE	        Meaning
    # > 0.05	    Strong chaos
    # 0.005 – 0.05	Weak chaos / noise
    # ≈ 0	        Periodic or quasiperiodic
    # < 0	        Convergent or stable
    # PSY25sc_5_S10.0__6469.0_19_44 isssss 0.0040970970813493345
    # 10V 7470 is 0.0017555066965576577

    # cd = nd.corr_dim(x, emb_dim=m)
    # print(cd)
    # Correlation dimension	    System type
    # ~1	                    limit cycle
    # ~2	                    torus
    # 2–3.5                     fractional	chaotic attractor
    # >>5	                    noise dominates
    # PSY25sc_5_S10.0__6469.0_19_44 isssss 3.01956313919753
    # 10V 7470 is 2.8598725534759226

    # fd = 6469.0  # driving frequency
    # dt = 1 / 250000  # sampling rate
    # if t[1] != dt:
    #     raise ValueError  # quits if the first timestep is not the same as the sample rate todo
    T = int(1 / (fd * dt))

    poincare = X[::T]

    plt.scatter(poincare[:, 0], poincare[:, 1], s=3)
    # plt.xlabel('x(t)')
    # plt.ylabel('x(t + τ)')
    plt.show()

    # work out when im in thesis pending
    # measure EVERYTHING (amplitude, frequency)


def refine_peak_time(t, y, i):
    # Cannot refine at edges
    if i <= 0 or i >= len(t) - 1:
        return t[i]

    # three points around peak
    x = t[i - 1:i + 2]
    Y = y[i - 1:i + 2]

    # quadratic fit
    a, b, c = np.polyfit(x, Y, 2)

    # vertex: x = -b/(2a)
    return -b / (2 * a)


def refine_peak_time_sinefit_old(t, F, i, window=5):
    """
    Refine peak time by fitting a local sinusoid to the driving force.

    t, F : data arrays
    i : coarse peak index from find_peaks
    window : number of points on each side of peak to fit (default = 5 → 11-point window)
    """

    # choose fitting window around the peak
    i0 = max(0, i - window)
    i1 = min(len(t), i + window + 1)

    t_win = t[i0:i1]
    F_win = F[i0:i1]

    # center time to improve numerical conditioning
    t0 = t[i]
    tt = t_win - t0

    # initial frequency guess from coarse period
    # (This improves stability if the sine frequency varies)
    # Estimate instantaneous frequency from t array around the peak:
    dt = np.mean(np.diff(t))
    # crude estimate: assume peak spacing roughly Npoints * dt
    # but we don't know next peak, so assume ω ~ π / dt
    omega_guess = np.pi / (dt * 5)  # fairly tolerant guess

    # sine model with no DC offset (optional to include offset)

    def model(t, A, omega, phi):
        return A * np.sin(omega * t + phi)

    # initial parameter guesses
    A0 = F[i]  # amplitude approx
    p0 = [A0, omega_guess, 0.0]

    # Fit the sinusoid
    try:
        popt, _ = curve_fit(model, tt, F_win, p0=p0, maxfev=5000)
        A, omega, phi = popt

        # compute peak time relative to t0:
        # peak occurs where derivative=0 → argument = π/2
        t_peak_local = (np.pi / 2 - phi) / omega

        # convert back to absolute time
        return t0 + t_peak_local

    except RuntimeError:
        # fallback: return the original coarse peak time
        return t[i]


def refine_peak_time_sinefit(t, F, i, window=4, freq_guess=float(6000)):
    """
    Robust peak-time refinement via linear sine+cosine fit.
    Avoids amplitude inversion.

    freq_guess: expected driving frequency in Hz.
    window: half-window size around peak (4 → 9 points).
    """

    # compute angular frequency
    omega = 2 * np.pi * float(freq_guess)

    # window for fitting
    i0 = max(0, i - window)
    i1 = min(len(t), i + window + 1)

    t_win = t[i0:i1]
    F_win = F[i0:i1]

    t0 = t[i]  # center time to stabilize numerics
    tt = t_win - t0

    # Construct design matrix for A*cos(ωt) + B*sin(ωt) + C
    X = np.column_stack([
        np.cos(omega * tt),
        np.sin(omega * tt),
        np.ones_like(tt)
    ])

    # linear least squares fit
    coef, _, _, _ = np.linalg.lstsq(X, F_win, rcond=None)
    A, B, C = coef

    # amplitude and phase
    phi = np.arctan2(B, A)  # phase of cosine+sine

    # peak occurs at cosine maximum → argument = 0
    # A*cos(ω(t - t0)) has max when (ω(t_peak - t0) + phi = 0)
    t_peak_local = -phi / omega

    return t0 + t_peak_local


def refine_peak_parabolic(t, F, i):
    """
    Analytic 3-point parabolic peak interpolation.
    Works extremely well for low-sample sinusoidal and slightly distorted peaks.
    """
    if i == 0 or i == len(F)-1:
        return t[i]

    y0 = F[i-1]
    y1 = F[i]
    y2 = F[i+1]

    denom = (y0 - 2*y1 + y2)
    if denom == 0:
        return t[i]

    delta = 0.5 * (y0 - y2) / denom   # offset in sample units
    dt = t[1] - t[0]

    return t[i] + delta * dt


def plotting_bifurc():

    spectra_path = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents\Shared - Mechanical " \
                   r"Vibrations of Ultrathin Films\Lab\data\PSY\Shapes\Semicircle\raw"
    files = os.listdir(spectra_path)
    files.sort()

    mode = 'bifurc'
    mode = 'individual'
    mode = 'fourier'
    removers = None
    # removers = 'frequency'
    removers = 'amplitude'
    only_this_amplitude = 10
    only_this_frequency = 6460

    if mode == 'birufc':
        fig, ax = plt.subplots()

    for i, file in tqdm(enumerate(files), total=len(files)):
        if file.split(".")[-1] != "txt":  # check it's a data file and not a folder or something else
            continue  # not a data file
        if len(file.split('__')[0].split('_')[-1]) == 1:
            # PSY25sc_5_S__6474.0_24_44.txt rather than PSY25sc_5_S9.0__6474.0_24_44.txt
            continue  # didn't note amplitude. recoverable later by looking at driving force amplitude data!

        ad = float(file.split('__')[0].split('_')[-1][1:])  # driving amplitude
        if ad != only_this_amplitude and removers == 'amplitude':
            continue  # ignore other amplitudes for now
        # time, driving force, sample, driving frequency, sampling rate
        # t, F, x, fd, dt = get_from_file(spectra_path + '\\' + file, num=int(1e3))
        t, F, x, fd, dt = get_from_file(spectra_path + '\\' + file)
        if fd != only_this_frequency and removers == 'frequency':
            continue  # ignore other driving frequencies for now

        peaks, _ = find_peaks(F)
        # t_true_peaks = np.array([refine_peak_time(t, F, i) for i in peaks])
        # t_true_peaks = np.array([refine_peak_time_sinefit(t, F, j, window=4, freq_guess=fd) for j in peaks])
        t_true_peaks = np.array([refine_peak_parabolic(t, F, j) for j in peaks])
        interpF = interp1d(t, F, kind='cubic')
        interpx = interp1d(t, x, kind='cubic')
        sample_at_peaks = interpx(t_true_peaks)
        sine_at_peaks = interpF(t_true_peaks)

        # plot it each cycle
        if mode == 'bifurc':
            ax.plot(np.ones_like(sample_at_peaks) * ad, sample_at_peaks, '.')
            ax.plot(np.ones_like(sample_at_peaks) * fd, sample_at_peaks, '.')
        else:
            fig, ax = plt.subplots()
            if mode == 'individual':
                ax.plot(t, F, label='driving force sampled')
                ax.plot(t, x, label='data sampled')
                ax.plot(t_true_peaks, sine_at_peaks, 'o', label='driving force peaks interpolated')
                ax.plot(t_true_peaks, sample_at_peaks, 'o', label='sample peaks interpolated')
            if mode == 'fourier':
                tlen = t[-1] - t[0]
                rate = (1 / dt)
                min_freq = 1 / tlen
                max_freq = rate / 2  # = (num / 2) / t  # aka Nyquist frequency
                num = np.ceil(rate * tlen)  # number of samples to measure
                freqs = np.linspace(start=min_freq, stop=max_freq, num=int(num / 2), endpoint=True)
                ax.plot(freqs, np.abs(np.fft.fft(F - np.mean(F)) ** 2)[1:int(num / 2) + 1], label='driving')
                ax.plot(freqs, np.abs(np.fft.fft(x - np.mean(x)) ** 2)[1:int(num / 2) + 1], label='response')
                # fourier transform compare normal (large amplitude) to weird ones (small amplitude),
                # see if there's presense of a low frequency
            plt.legend()
            plt.show()

    print('done')
    if mode == 'bifurc':
        plt.xlabel('frequency / Hz')
        plt.ylabel(f'amplitude at {ad}V peak driving / V')
        plt.xlabel('amplitude / V')
        plt.ylabel(f'amplitude at {fd}Hz driving / V')
        plt.show()

