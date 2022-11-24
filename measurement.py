import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter  # smoothing
import nidaqmx
from tqdm import tqdm
import time

from my_tools import ax_lims
from fitting import fit_fast
from IO_setup import set_up_signal_generator_sine, set_up_signal_generator_pulse


def measure(freq=None, freqstep=5, t=2, suppressed=False):
    """
    Measures a film by exciting a series of frequencies using sine waves then measuring the response.
    freq=[minf, maxf] is the minimim and maximum frequencies to sweep between
    freqstep=df is the frequency step between measurements
    t=t is the length of time a measurement takes
    """
    if freq is None:
        freq = [50, 2000]

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

    rate = 10001
    num = int(rate * t)
    times = np.arange(num) / rate
    line_current, = ax_current.plot(times, np.zeros_like(times))
    ax_current.plot([times[0], times[-1]], [0, 0], 'k--')
    ax_current.set_ylim((-12, 12))
    line_all, = ax_all.plot([0, 0], [0, 0], label='Data')

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        # print(chan.ai_rng_high)

        with open(f"outputs/output.txt", 'w') as out:
            num_freqs = 1 + int(np.abs(np.diff(freq)) / freqstep)
            data_list = np.zeros(num_freqs)
            freqs = np.linspace(freq[0], freq[1], num_freqs)
            for i, f in tqdm(enumerate(freqs), total=len(freqs), ncols=100):
                # set current frequency
                sig_gen.write(f'APPLy:SINusoid {f}, 5')  # todo !!! changed from 10 to 5 Vpp

                # read the microphone data after a short pause
                time.sleep(min(t / 10, 0.05))
                signal = task.read(num)

                # process and write to file
                data = np.sqrt(np.mean(np.square(signal)))  # calculate RMS of signal
                data_list[i] = data
                ax_current.set_title(f"Frequency: {f:.6g}, Response: {data:.3g}")
                out.write(f"{f:.6g} {data:.6g}\n")

                # update visual plot of data
                line_current.set_ydata(signal)
                if i > 0:
                    indexes = data_list.nonzero()
                    line_all.set_xdata(freqs[indexes])
                    line_all.set_ydata(data_list[indexes])
                    if i >= int(len(freqs) * 0.75):
                        if i == int(len(freqs) * 0.75):
                            # plot a fit line
                            line_all_fit, = ax_all.plot([0, 0], [0, 0], label='Lorentzian fit')
                            ax_all.legend()
                        fity, values = fit_fast(freqs[indexes], data_list[indexes])
                        line_all_fit.set_xdata(freqs[indexes])
                        line_all_fit.set_ydata(fity)
                        ax_all.set_title(f"Response with fit: gamma={values[0]:.3g}, x0={values[1]:.4g}, "
                                         f"c={values[2]:.3g}, a={values[3]:.3g}")
                    ax_all.set_ylim((0, ax_lims(data_list[indexes])[1]))
                fig.canvas.draw()
                fig.canvas.flush_events()
    sig_gen.write('OUTPut OFF')  # stop making an annoying noise!
    plt.close(fig)
    plt.ioff()
    print(f"Lorentzian fit x0 = {values[1]}")
    print(f"Maximum peak value at f = {freqs[np.argmax(data_list)]}")

    if suppressed:
        return freqs, data_list


def measure_pulse_decay(devchan="Dev1/ai0"):
    """
    Measures a film by hitting it with a short pulse and measuring the response.
    freq=[minf, maxf] is the minimim and maximum frequencies to measure
    """

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
    runs = int(100)
    t = 0.2
    sleep_time = 0.135
    rate = 10001  # max is 250,000 samples per second >:D
    num = int(np.ceil(rate * t))  # number of samples to measure
    times = np.arange(start=0, stop=t, step=(1 / rate))
    line_current, = ax_current.plot(times, np.zeros_like(times), label="Raw Data")

    # setting up frequency variables
    min_freq = 1 / t
    max_freq = rate / 2  # = (num / 2) / t  # aka Nyquist frequency
    # num_freqs = (max_freq - 0) / min_freq  # is this wrong?
    freqs = np.linspace(start=min_freq, stop=max_freq, num=int((num / 2) - 1), endpoint=True)
    line_all_current, = ax_all.plot(freqs, np.zeros_like(freqs), label="Previous")
    line_all, = ax_all.plot(freqs, np.zeros_like(freqs), label="Moving Average")
    ax_all.legend()
    ax_current.plot([times[0], times[-1]], [0, 0], 'k--', label="_Zero line")
    ax_current.plot([t / 3, t / 3], [-12, 12], 'k:', label="_Time Boundary")
    ax_all.set_xlim(ax_lims([min_freq, max_freq]))
    ax_current.set_xlim([times[0], times[-1]])
    ax_current.set_ylim((-12, 12))

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(devchan, min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        data_list = np.ones([len(freqs), runs]) * np.nan
        response = data = y = np.zeros_like(data_list[:, 0])  # make room in memory
        for i in tqdm(range(runs), total=runs, ncols=100):
            complete = False
            while not complete:
                # reset signal generator output to get to a known timing
                sig_gen.write(f'APPLy:PULSe {10}, MAX')
                sig_gen.write(f'APPLy:PULSe {1 / t}, MAX')
                ax_current.set_title(f"Run number: {str(i).zfill(len(str(runs)))} of {runs}")
                start = time.time()
                if i > 0:
                    # discount the pulse and everything before it
                    response = np.where(range(len(signal)) > np.argmax(np.abs(signal)) + 10, signal, 0)
                    # process and store
                    data = np.abs(np.fft.fft(response - np.mean(response))[1:int(num / 2)])
                    data_list[:, i - 1] = data

                    # update visual plot of data
                    line_current.set_ydata(response)
                    line_all_current.set_ydata(data)
                    # ax_current.set_ylim(ax_lims(signal))
                    y = np.nanmean(data_list, 1)
                    line_all.set_ydata(y)
                    ax_all.set_ylim((0, ax_lims(y)[1]))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                elif i == 0:
                    time.sleep(sleep_time)
                if sleep_time - (time.time() - start) > 1e-3:
                    time.sleep(sleep_time - (time.time() - start))  # wait for next cycle
                else:
                    # wait for the cycle after the next cycle
                    sig_gen.write(f'APPLy:PULSe {10}, MAX')
                    sig_gen.write(f'APPLy:PULSe {1 / t}, MAX')
                    time.sleep(sleep_time)
                # read the raw microphone signal
                signal = task.read(num)
                if np.argmax(np.abs(signal)) < len(signal) / 3:
                    complete = True
                # else:
                #     print(f"Repeating iteration {i}")  # this print is annoying
        # process and store
        response = np.where(range(len(signal)) > np.argmax(np.abs(signal)) + 5, signal, 0)  # discount the pulse
        data_list[:, i] = np.abs(np.fft.fft(response - np.mean(response))[1:int(num / 2)])

    sig_gen.write('OUTPut OFF')  # stop making an annoying noise!
    plt.close(fig)
    plt.ioff()

    # file management
    arr = np.zeros([len(freqs), 2])
    arr[:, 0] = freqs
    arr[:, 1] = np.nanmean(data_list, 1) / np.mean(np.nanmean(data_list, 1))  # somewhat normalise (mean = 1)
    np.savetxt("outputs/output.txt", arr, fmt='%.6g')
