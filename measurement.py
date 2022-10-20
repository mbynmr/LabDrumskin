import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # smoothing
import nidaqmx
from tqdm import tqdm
import time


from my_tools import ax_lims, resave_output
from IO_setup import set_up_signal_generator_sine, set_up_signal_generator_pulse


def measure(freq=None, freqstep=5, t=2):
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
    ax_current.set_title(f"Current Frequency: {freq[0]:.6g}")
    ax_all.set_ylabel("Mean(Abs(Amplitude)) / V")
    ax_all.set_xlabel("Frequency / Hz")
    ax_all.set_title(f"Response")
    plt.tight_layout()

    rate = 2000  # default 1000? It is not. Could find it out somehow
    num = int(rate * t)
    times = np.arange(num) / rate
    line_current, = ax_current.plot(times, np.zeros_like(times))
    line_all, = ax_all.plot([0, 0], [0, 0])

    fig.canvas.draw()
    fig.canvas.flush_events()
    # time.sleep(0.5)  # just to chill after you press start for a second

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        with open(f"outputs/output.txt", 'w') as out:
            num_freqs = 1 + int((freq[1] - freq[0]) / freqstep)
            data_list = np.zeros(num_freqs)
            freqs = np.linspace(freq[0], freq[1], num_freqs)
            for i, f in tqdm(enumerate(freqs)):
                sig_gen.write(f'APPLy:SINusoid {f}, 10')
                ax_current.set_title(f"Frequency: {f:.6g}")

                # read the microphone data
                signal = task.read(num)

                # process and write to file
                data = np.mean(np.abs(signal))  # todo check if it is mean(abs()) ?
                data_list[i] = data
                out.write(f"{f:.6g} {data:.6g}\n")

                # update visual plot of data
                line_current.set_ydata(signal)
                ax_current.set_ylim(ax_lims(signal))

                if i > 0:
                    line_all.set_xdata(freqs[np.nonzero(data_list)])
                    line_all.set_ydata(data_list[np.nonzero(data_list)])
                    ax_all.set_ylim((0, ax_lims(data_list[np.nonzero(data_list)])[1]))
                else:
                    ax_all.set_xlim([freq[0], freq[1]])
                    ax_current.set_xlim([0, t])
                fig.canvas.draw()
                fig.canvas.flush_events()
    plt.close(fig)

    # file management
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
    ax_all.set_ylabel("Mean(Abs(Amplitude)) / V")
    ax_all.set_xlabel("Frequency / Hz")
    ax_all.set_title(f"Average Response")
    plt.tight_layout()

    # setting up data collection variables
    runs = int(10)
    t = 5  # todo response for 5 secs
    rate = 4000  # max is 250,000 samples per second >:D
    num = int(rate * t)  # number of samples to measure
    times = np.arange(start=0, stop=t, step=(1 / rate))
    line_current, = ax_current.plot(times, np.zeros_like(times))

    # setting up frequency variables
    min_freq = 1 / t
    max_freq = rate / 2  # = (num / 2) / t  # aka Nyquist frequency
    freqs = np.linspace(start=min_freq, stop=max_freq, num=int((num - 1) / 2), endpoint=True)
    line_all, = ax_all.plot(freqs, np.zeros_like(freqs))
    ax_all.set_xlim(ax_lims([min_freq, max_freq]))
    ax_current.set_xlim([0, t])

    fig.canvas.draw()
    fig.canvas.flush_events()
    # time.sleep(0.5)  # just to chill after you press start for a second

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        data_list = np.ones([len(freqs), runs]) * np.nan
        for i in tqdm(range(runs)):
            # sig_gen.write(f'APPLy:SINusoid {f}, 10')
            # todo do 1 pulse then measure, so we don't have to see the pulse, just the remaining resonance.
            ax_current.set_title(f"Run number: {str(i).zfill(len(str(runs)))}")

            # sig_gen.write(f'APPLy:PULSe {0.25 + 2 * np.random.random()}, MAX')  # randomise signal duration
            sig_gen.write(f'APPLy:PULSe {5/3}, MAX')  # randomise signal duration
            signal = task.read(num)  # raw signal

            # process and write to file
            data = np.abs(np.fft.fft(signal)[1:int(num / 2)])
            data_list[:, i] = data
            # update visual plot of data
            line_current.set_ydata(signal)
            ax_current.set_ylim(ax_lims(signal))
            y = np.nanmean(data_list, 1)
            # y = savgol_filter(y, int(num / 1000), 3)  # order of fitted polynomial
            line_all.set_ydata(y)
            ax_all.set_ylim((0, ax_lims(y)[1]))
            fig.canvas.draw()
            fig.canvas.flush_events()
    plt.close(fig)

    data_out = np.nanmean(data_list, 1)
    # data_out = savgol_filter(data_out, int(num / 1000), 3)  # todo smoothed, watch out!
    data_out = data_out / np.mean(data_out)  # somewhat normalise so the average value is 1 (not maximum)

    # file management
    with open(f"outputs/output.txt", 'w') as out:
        for i, f in enumerate(freqs):
            out.write(f"{f:.6g} {data_out[i]:.6g}\n")
    resave_output()