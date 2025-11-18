import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter  # smoothing
import nidaqmx as ni
from tqdm import tqdm
import time
from noisyopt import minimizeCompass
import sys

from my_tools import ax_lims, copy2clip, round_sig_figs
from fitting import fit_fast
from IO_setup import set_up_signal_generator_sine, set_up_signal_generator_pulse


def measure_sweep(freq=None, freqstep=5, t=2, suppressed=False, vpp=10, devchan="Dev1/ai0", GUI=None,
                  save_path=None, temp=None, sample=None):
    """
    Measures a film by exciting a series of frequencies using sine waves then measuring the response.
    freq=[minf, maxf] is the minimim and maximum frequencies to sweep between
    freqstep=df is the frequency step between measurements
    t=t is the length of time a measurement takes
    """
    if freq is None:
        freq = [50, 4000]

    if freqstep == 0:  # automatic frequency step selection
        freqstep = (freq[0] - freq[1]) / 300
        freqstep = round_sig_figs(freqstep, 2, method='d')

    if GUI is None:
        printer = sys.stderr
        pauser = Pauser
    else:
        printer = GUI.Writer
        pauser = GUI

    sig_gen = set_up_signal_generator_sine()

    plt.ion()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax_all, ax_current = axs
    ax_current.set_ylabel("Amplitude / V")
    ax_current.set_xlabel("time / s")
    ax_current.set_title(f"Frequency: {freq[0]:.6g}")
    ax_all.set_ylabel("RMS Voltage / V")
    ax_all.set_xlabel("Frequency / Hz")
    ax_all.set_title(f"Response Spectra")
    ax_all.set_xlim([freq[0], freq[1]])
    ax_current.set_xlim([0, t])
    plt.tight_layout()

    rate = 20000
    num = int(rate * t)
    times = np.arange(num) / rate
    line_current, = ax_current.plot(times, np.zeros_like(times))
    ax_current.plot([times[0], times[-1]], [0, 0], 'k--')
    ax_current.set_ylim((-12, 12))
    line_all, = ax_all.plot([0, 0], [0, 0], label='Data')

    with ni.Task() as task:
        task.ai_channels.add_ai_voltage_chan(devchan, min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        if save_path is not None:
            task.ai_channels.add_ai_voltage_chan("Dev2/ai2", min_val=-10.0, max_val=10.0)
            task.timing.cfg_samp_clk_timing(rate=rate / 2, samps_per_chan=num)
        # print(chan.ai_rng_high)

        with open(f"outputs/output.txt", 'w') as out:
            num_freqs = 1 + int(np.abs(np.diff(freq)) / freqstep)
            data_list = np.zeros(num_freqs)
            freqs = np.linspace(freq[0], freq[1], num_freqs)
            a = np.array(freqs)
            np.random.default_rng().shuffle(a)
            for i, f in tqdm(enumerate(freqs), total=len(freqs), ncols=52, file=printer):

                # pause control
                if pauser.pause.get():
                    pauser.w.wait_variable(pauser.pause)
                    # pauser.pause.set(False)

                # set current frequency
                sig_gen.write(f'APPLy:SINusoid {f}, {vpp}')

                # read the microphone data after a short pause
                time.sleep(0.1)
                signal = task.read(num)  # todo this is now twice as big?

                # save raw
                if save_path is None:
                    raw = np.zeros([len(signal), 2])
                    raw[:, 0] = np.linspace(0, t, num=len(signal))
                    raw[:, 1] = signal
                    np.savetxt("outputs/raw.txt", raw, fmt='%.4g')
                else:
                    raw = np.zeros([len(signal[0]), 3])
                    raw[:, 0] = np.linspace(0, t, num=len(signal[0]))
                    raw[:, 1] = signal[0]
                    raw[:, 2] = signal[1]
                    # print(save_path + f"/raw/{sample}_S__{f}_{i}_{temp}.txt")
                    np.savetxt(save_path + f"/raw/{sample}_S__{f}_{i}_{temp:.2g}.txt", raw, fmt='%.4g')
                    # save_all_raw(raw, f, 'S', save_path, temp, sample)

                # todo
                #  measure from 1 daq card the mic response and sine wave into piezo, line up timings and save together.
                #  plot timed lines together and have a cheeky looksy
                #  plot driving amplitude vs poincare section for bifurcation
                # todo measure some more thicknesses on elip

                # process and write to file
                data = np.sqrt(np.mean(np.square(signal)))  # calculate RMS of signal
                data_list[np.argwhere(freqs == f)] = data
                ax_current.set_title(f"Frequency: {f:.6g}, Response: {data:.3g}")
                out.write(f"{f:.6g} {data:.6g}\n")

                # update visual plot of data

                # smaller to plot
                # line_current.set_ydata(signal)
                indexes_for_speedy_plot = [np.sort(np.random.choice(len(signal), int(0.1 * len(signal))))]
                if save_path is not None:
                    line_current.set_ydata(np.array(signal)[indexes_for_speedy_plot])
                    line_current.set_xdata(np.array(times)[indexes_for_speedy_plot])

                if i > 0:
                    indexes = data_list.nonzero()
                    line_all.set_xdata(freqs[indexes])
                    line_all.set_ydata(data_list[indexes])
                    if i >= int(len(freqs) * 0.95):
                        if i == int(len(freqs) * 0.95):
                            # plot a fit line
                            line_all_fit, = ax_all.plot([0, 0], [0, 0], label='Lorentzian fit')
                            ax_all.legend()
                        fity, values = fit_fast(freqs[indexes], data_list[indexes])
                        line_all_fit.set_xdata(freqs[indexes])
                        line_all_fit.set_ydata(fity)

                        # raw = np.zeros([len(signal), 2])
                        # raw[:, 0] = np.linspace(0, 0.2, num=len(signal))
                        # raw[:, 1] = signal
                        # # raw[:, 2] = np.sin(1e3 / (2 * np.pi) * np.linspace(0, 0.2, num=len(signal)))
                        # np.savetxt("outputs/raw.txt", raw, fmt='%.4g')

                        ax_all.set_title(f"Response with fit: gamma={values[0]:.3g}, x0={values[1]:.4g}, "
                                         f"c={values[2]:.3g}, a={values[3]:.3g}")
                    ax_all.set_ylim((0, ax_lims(data_list[indexes])[1]))
                fig.canvas.draw()
                fig.canvas.flush_events()
    sig_gen.write('OUTPut OFF')  # stop making an annoying noise!
    plt.close(fig)
    plt.ioff()
    # print(f"Lorentzian fit x0 = {values[1]}")
    # print(f"Maximum peak value at f = {freqs[np.argmax(data_list)]}")

    if suppressed:
        return freqs, data_list


def measure_pulse_decay(devchan="Dev1/ai0", runs=100, delay=20, t=0.2, GUI=None):
    """
    Measures a film by hitting it with a short pulse and measuring the response
    """
    if GUI is None:
        printer = sys.stderr
        pauser = Pauser
    else:
        printer = GUI.Writer
        pauser = GUI

    sig_gen = set_up_signal_generator_pulse()

    # setting up plots
    plt.ion()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax_all, ax_current = axs
    ax_current.set_ylabel("Power")
    ax_current.set_xlabel("time / s")
    ax_current.set_title(f"Run number: {-1:.6g}")
    ax_all.set_ylabel("Response")
    ax_all.set_xlabel("Frequency / Hz")
    ax_all.set_title(f"Response Spectra")
    plt.tight_layout()

    # setting up data collection variables
    runs = int(runs)
    sleep_time = 0.135
    rate = 20000  # 20e3
    num = int(np.ceil(rate * t))  # number of samples to measure
    times = np.arange(start=0, stop=t, step=(1 / rate))
    line_current, = ax_current.plot(times, np.zeros_like(times), label="Raw Data")

    # setting up frequency variables
    min_freq = 1 / t
    max_freq = rate / 2  # = (num / 2) / t  # aka Nyquist frequency
    # num_freqs = (max_freq - 0) / min_freq  # is this wrong?
    freqs = np.linspace(start=min_freq, stop=max_freq, num=int(num / 2), endpoint=True)
    line_all_current, = ax_all.plot(freqs, np.zeros_like(freqs), label="Previous")
    line_all, = ax_all.plot(freqs, np.zeros_like(freqs), label="Moving Average")
    # line_all_min, = ax_all.plot(freqs, np.zeros_like(freqs), label="Minimum")
    # line_all_max, = ax_all.plot(freqs, np.zeros_like(freqs), label="Maximum")
    ax_all.legend(loc='upper left')
    ax_current.plot([times[0], times[-1]], [0, 0], 'k--', label="_Zero line")
    ax_current.plot([t / 3, t / 3], [-12, 12], 'k:', label="_Time Boundary")
    ax_all.set_xlim(ax_lims([min_freq, max_freq]))
    ax_current.set_xlim([times[0], times[-1]])
    ax_current.set_ylim((-12, 12))
    plt.tight_layout()

    with ni.Task() as task:
        task.ai_channels.add_ai_voltage_chan(devchan, min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        data_list = np.ones([len(freqs), runs]) * np.nan
        datar_list = np.ones([len(freqs), runs]) * np.nan
        datai_list = np.ones([len(freqs), runs]) * np.nan
        # response = data = y = np.zeros_like(freqs)  # make room in memory
        for i in tqdm(range(runs + 1), total=runs + 1, ncols=52, file=printer):
            complete = False
            process = True
            while not complete:

                # pause control
                if pauser.pause.get():
                    pauser.w.wait_variable(pauser.pause)
                    # pauser.pause.set(False)

                # reset signal generator output to get to a known timing
                sig_gen.write(f'APPLy:PULSe {10}, MAX')
                sig_gen.write(f'APPLy:PULSe {1 / t}, MAX')
                ax_current.set_title(f"Run number: {str(i).zfill(len(str(runs)))} of {runs}")
                start = time.time()
                if i > 0 and process:  # do processing of previous signal
                    response = signal

                    # process and store
                    # todo this all matters a lot i think, especially for the piezo peak.
                    fft = np.fft.fft(response - np.mean(response))
                    data = np.abs(fft[1:int(num / 2) + 1])  # POWER SPECTRA (well, almost, it needs to be squared)
                    datar = np.real(fft[1:int(num / 2) + 1])  # real
                    datai = np.imag(fft[1:int(num / 2) + 1])  # imaginary
                    data_list[:, i - 1] = data
                    datar_list[:, i - 1] = datar
                    datai_list[:, i - 1] = datai

                    # update visual plot of data
                    line_current.set_ydata(signal)
                    line_all_current.set_ydata(data)

                    # save raw
                    raw = np.zeros([len(signal), 2])
                    raw[:, 0] = np.linspace(0, t, num=len(signal))
                    raw[:, 1] = signal
                    np.savetxt("outputs/raw.txt", raw, fmt='%.4g')

                    # line_all_min.set_ydata(np.nanmin(data_list, 1))
                    # line_all_max.set_ydata(np.nanmax(data_list, 1))
                    y = np.nanmean(data_list, 1)
                    line_all.set_ydata(y)
                    ax_all.set_ylim((0, ax_lims(y)[1]))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    process = False
                # elif i == 0:
                #     time.sleep(sleep_time)
                if sleep_time - (time.time() - start) > 1e-3 + 0.2e-3:
                    time.sleep(sleep_time - (time.time() - start))  # wait for next cycle
                else:
                    # wait for the cycle after the next cycle
                    sig_gen.write(f'APPLy:PULSe {10}, MAX')
                    sig_gen.write(f'APPLy:PULSe {1 / t}, MAX')
                    time.sleep(sleep_time)  # todo do i need this sleep?
                # read the raw microphone signal
                signal = task.read(num)
                if np.argmax(np.abs(signal)) < len(signal) / 3:
                    complete = True
                # else:
                #     print(f"Repeating iteration {i}")  # this print is annoying

    sig_gen.write('OUTPut OFF')  # stop making an annoying noise!
    plt.close(fig)
    plt.ioff()
    # print(f"Maximum peak value at f = {freqs[np.argmax(data_list)]}")

    # file management
    arr = np.zeros([len(freqs), 4])
    arr[:, 0] = freqs
    arr[:, 1] = np.nanmean(data_list ** 2, 1)  # NON-normalised  # 2025/04/02 big change: put in the ** 2 operation
    arr[:, 2] = np.nanmean(datar_list, 1)
    arr[:, 3] = np.nanmean(datai_list, 1)
    np.savetxt("outputs/output.txt", arr, fmt='%.6g')


def measure_adaptive(devchan="Dev2/ai0", vpp=5, tolerance=5, start_guess=1e3, deltainit=1e3, bounds=None):
    if bounds is None:
        bounds = [100, 4e3]
    m = Measure(t=0.2, vpp=vpp, devchan=devchan)
    res = minimizeCompass(m.measure, x0=[start_guess], bounds=[bounds], errorcontrol=True, disp=False, paired=False,
                          deltainit=1e3, deltatol=tolerance, funcNinit=4, funcmultfactor=1.25)
    m.close()
    print(f"{res.x[0] = }")
    copy2clip(f"{res.x[0]:.6g}")


def save_all_raw(raw, method, save_path, temperature, sample):
    # todo automate amplitude changing. last thing to automate
    # method=method, save_path=self.save_path.get(), temperature=self.temp.get(), sample=self.sample_name.get()
    np.savetxt(save_path + f"/{sample}_{method}_{temperature}.txt", raw, fmt='%.4g')


class Measure:
    def __init__(self, t, vpp=5, devchan="Dev2/ai0"):
        rate = 10001
        self.num = int(rate * t)
        # self.times = np.arange(self.num) / self.rate
        self.task = ni.Task()
        self.task.ai_channels.add_ai_voltage_chan(devchan, min_val=-10.0, max_val=10.0)
        self.task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=self.num)
        self.sig_gen = set_up_signal_generator_sine()
        self.vpp = vpp
        self.output = open("outputs/output.txt", "w")

        # plotting things
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Frequency / Hz")
        self.ax.set_ylabel("Response")
        self.ax.set_title("Testing f = ")
        self.line, = self.ax.plot([0, 0], [0, 0], '.')
        self.ax.set_xlim([0, 4e3])
        self.x = []
        self.y = []

    def measure(self, f):
        self.sig_gen.write(f'APPLy:SINusoid {float(f)}, {self.vpp}')  # set the signal generator to frequency f
        time.sleep(0.05)
        rms = np.sqrt(np.mean(np.square(self.task.read(self.num))))  # read signal from microphone then calculate RMS
        self.output.write(f"{float(f):.6g} {rms:.6g}\n")

        # plotting things
        self.x.append(float(f))
        self.y.append(rms)
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
        self.ax.set_ylim((0, ax_lims(self.y)[1]))
        self.ax.set_title(f"Testing f = {float(f):.6g}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return -rms  # function is minimisation so return negative of signal value

    def close(self):
        plt.close(self.fig)
        plt.ioff()
        self.sig_gen.write('OUTPut OFF')
        self.task.close()
        self.sig_gen.close()
        self.output.close()


class Pauser:
    class pause:
        @classmethod
        def get(cls):
            return False
        @classmethod
        def set(cls, _):
            pass

    def wait_variable(self):
        pass
