import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter  # smoothing
import nidaqmx
from tqdm import tqdm
import time

from my_tools import ax_lims, resave_output
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
    # if suppressed:
    #     fig.set_visible(False)
    # else:
    #     fig.set_visible(True)
    ax_all, ax_current = axs
    ax_current.set_ylabel("Amplitude / V")
    ax_current.set_xlabel("time / s")
    ax_current.set_title(f"Frequency: {freq[0]:.6g}")
    ax_all.set_ylabel("RMS Voltage / V")
    ax_all.set_xlabel("Frequency / Hz")
    ax_all.set_title(f"Response")
    ax_all.set_xlim([freq[0], freq[1]])
    ax_current.set_xlim([0, t])
    plt.tight_layout()

    rate = 5000  # default 1000? It is not. Could find it out somehow
    num = int(rate * t)
    times = np.arange(num) / rate
    line_current, = ax_current.plot(times, np.zeros_like(times))
    line_all, = ax_all.plot([0, 0], [0, 0])

    with nidaqmx.Task() as task:
        chan = task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        print(chan.ai_rng_high)

        with open(f"outputs/output.txt", 'w') as out:
            out.write(f"Frequency Amplitude")
            num_freqs = 1 + int(np.diff(freq) / freqstep)
            data_list = np.zeros(num_freqs)
            freqs = np.linspace(freq[0], freq[1], num_freqs)
            for i, f in tqdm(enumerate(freqs)):
                sig_gen.write(f'APPLy:SINusoid {f}, 10')
                ax_current.set_title(f"Frequency: {f:.6g}")

                # read the microphone data
                signal = task.read(num)
                # process and write to file
                data = np.sqrt(np.mean(np.square(signal)))  # calculate RMS of signal
                data_list[i] = data
                out.write(f"{f:.6g} {data:.6g}\n")

                # update visual plot of data
                line_current.set_ydata(signal)
                ax_current.set_ylim(ax_lims(signal))
                if i > 0:
                    line_all.set_xdata(freqs[np.nonzero(data_list)])
                    line_all.set_ydata(data_list[np.nonzero(data_list)])
                    ax_all.set_ylim((0, ax_lims(data_list[np.nonzero(data_list)])[1]))
                fig.canvas.draw()
                fig.canvas.flush_events()
    plt.close(fig)

    # file management
    if suppressed:
        return freqs, data_list
    else:
        resave_output()


def measure_pulse(freq=None):
    """
    Measures a film by hitting it with a short pulse and measuring the response.
    freq=[minf, maxf] is the minimim and maximum frequencies to measure
    """

    if freq is None:
        freq = [50, 2000]

    sig_gen = set_up_signal_generator_pulse()

    # setting up plots
    plt.ion()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax_all, ax_current = axs
    ax_current.set_ylabel("Power")
    ax_current.set_xlabel("time / s")
    ax_current.set_title(f"Run number: {-1:.6g}")
    ax_all.set_ylabel("RMS Voltage / V")
    ax_all.set_xlabel("Frequency / Hz")
    ax_all.set_title(f"Response")
    plt.tight_layout()

    # setting up data collection variables
    runs = int(100)
    t = 0.4
    rate = 4000  # max is 250,000 samples per second >:D
    num = int(rate * t)  # number of samples to measure
    times = np.arange(start=0, stop=t, step=(1 / rate))
    line_current, = ax_current.plot(times, np.zeros_like(times))

    # setting up frequency variables
    min_freq = 1 / t
    max_freq = rate / 2  # = (num / 2) / t  # aka Nyquist frequency
    freqs = np.linspace(start=min_freq, stop=max_freq, num=int((num - 1) / 2), endpoint=True)
    line_all, = ax_all.plot(freqs, np.zeros_like(freqs))
    line_all_current, = ax_all.plot(freqs, np.zeros_like(freqs))
    ax_all.set_xlim(ax_lims([min_freq, max_freq]))
    ax_current.set_xlim([0, t])

    with nidaqmx.Task() as task:
        chan = task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        data_list = np.ones([len(freqs), runs]) * np.nan

        for i in tqdm(range(runs)):
            # sig_gen.write(f'APPLy:SINusoid {f}, 10')
            # todo do 1 pulse then measure, so we don't have to see the pulse, just the remaining resonance.

            p = 1 / (0.1 * t + 0.5 * t * np.random.random())  # randomise signal duration
            ax_current.set_title(f"Run number: {str(i).zfill(len(str(runs)))}, Pulse Frequency: {p:.3g} Hz")
            sig_gen.write(f'APPLy:PULSe {p}, MAX')  # set signal duration
            # time.sleep(0.0001)
            signal = task.read(num)  # read the raw microphone signal
            # time.sleep(2)

            # process and write to file
            data = np.abs(np.fft.fft(signal - np.mean(signal))[1:int(num / 2)])
            data_list[:, i] = data
            # update visual plot of data
            line_current.set_ydata(signal)
            line_all_current.set_ydata(data)
            ax_current.set_ylim(ax_lims(signal))
            y = np.nanmean(data_list, 1)
            # y = savgol_filter(y, int(num / 1000), 3)  # order of fitted polynomial
            line_all.set_ydata(y)
            ax_all.set_ylim((0, ax_lims(y)[1]))
            fig.canvas.draw()
            fig.canvas.flush_events()
    plt.close(fig)

    data_out = np.nanmean(data_list, 1)
    # data_out = savgol_filter(data_out, int(num / 1000), 3)  # smoothed, watch out!
    data_out = data_out / np.mean(data_out)  # somewhat normalise so the average value is 1 (not maximum)

    # file management
    with open(f"outputs/output.txt", 'w') as out:
        out.write(f"Frequency Amplitude")
        for i, f in enumerate(freqs):
            out.write(f"{f:.6g} {data_out[i]:.6g}\n")
    resave_output()


def measure_adaptive(freq=None, t=2):
    """
    Measures a film by exciting a series of frequencies using sine waves then measuring the response.
    freq=[minf, maxf] is the minimim and maximum frequencies the peak is expected to be within
    t=t is the length of time a measurement takes
    This function chooses which frequency to measure based on the previous responses to find the peak quickly
    """

    sig_gen = set_up_signal_generator_sine()

    plt.ion()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax_all, ax_current = axs
    ax_current.set_ylabel("Amplitude / V")
    ax_current.set_xlabel("time / s")
    ax_current.set_title(f"Frequency:")
    ax_all.set_ylabel("RMS Voltage / V")
    ax_all.set_xlabel("Frequency / Hz")
    ax_all.set_title(f"Response")
    ax_all.set_xlim([freq[0], freq[1]])
    ax_current.set_xlim([0, t])
    plt.tight_layout()

    rate = 5000  # default 1000? It is not. Could find it out somehow
    num = int(rate * t)
    times = np.arange(num) / rate
    line_current, = ax_current.plot(times, np.zeros_like(times))
    line_all, = ax_all.plot([0, 0], [0, 0])

    with nidaqmx.Task() as task:
        chan = task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        with open(f"outputs/output.txt", 'w') as out:
            out.write(f"Frequency Amplitude")
            # num_freqs = 1 + int((freq[1] - freq[0]) / freqstep)
            # data_list = np.zeros(num_freqs)
            # freqs = np.linspace(freq[0], freq[1], num_freqs)
            # for i, f in tqdm(enumerate(freqs)):

            start = int(10)
            print(f"Finding {start} initial measurements")
            freqs, data_list = measure(freq=freq, freqstep=(np.diff(freq) / (start - 1)), t=t/10, suppressed=True)
            # todo measure in this function rather than suppress!
            print("Adapting...")  # adaptive
            found = False
            i = 0
            while not found:
                # sig_gen.write(f'APPLy:SINusoid {f}, 10')
                # ax_current.set_title(f"Frequency: {f:.6g}")
                f = np.mean(freqs[1] - freqs[0])  # test in between frequencies
                # f = np.random.random(1) * (np.amax(freqs) - np.amin(freqs)) + np.amin(freqs)
                print(f)#todo wtf
                sig_gen.write(f'APPLy:SINusoid {1}, 10')
                ax_current.set_title(f"Frequency: {f:.6g}")

                # read the microphone data
                signal = task.read(num)

                # process and write to file
                # data = np.mean(np.abs(signal))  # it is NOT mean(abs()), it is RMS
                data = np.sqrt(np.mean(np.square(signal)))  # calculate RMS
                data_list = np.array([*data_list, data])
                freqs = np.array([*freqs, f])
                out.write(f"{f:.6g} {data:.6g}\n")

                # update visual plot of data
                line_current.set_ydata(signal)
                ax_current.set_ylim(ax_lims(signal))
                line_all.set_xdata(freqs)
                line_all.set_ydata(data_list)
                ax_all.set_ylim((0, ax_lims(data_list)[1]))
                fig.canvas.draw()
                fig.canvas.flush_events()

                i += 1
                if i > 100:
                    print("No peak found.")
                    break
    plt.close(fig)

    # file management
    if found:
        resave_output()
