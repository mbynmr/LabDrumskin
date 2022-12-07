import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
# from tqdm import tqdm
import time
from scipy.optimize import curve_fit

from my_tools import resave_output, resave_auto, ax_lims
from fitting import lorentzian
from IO_setup import set_up_signal_generator_pulse  # , set_up_signal_generator_sine
# from measurement import measure, measure_adaptive, measure_pulse_decay


def temp_get(voltage):  # processes an array of voltages to return the corresponding array of temps (can be len = 1)
    zero = 0.055  # is probably 1 degree, and is from 0.051 to 0.055 pretty much
    hundred = -3.44
    # fifty = -1.62  # approximately
    # eighty_five = -2.88  # approximately
    # ninety_six = -3.24
    # temp = (100 / (hundred - zero)) * voltage - zero
    return (100 / (hundred - zero)) * np.asarray(voltage) - zero


class AutoTemp:
    """
    Automatically measures at given temperatures

    files will be saved for each temperature that is measured, and a final file that saves all the peaks and their temps
    """

    def __init__(self, save_folder_path, dev_signal, dev_temp, sample_name=None):
        self.save_folder_path = save_folder_path
        if sample_name is None:
            self.sample_name = input("Sample name:")
        else:
            self.sample_name = sample_name

        # setting up data collection variables
        self.runs = int(100)
        self.t = 0.2
        self.sleep_time = 0.135
        rate = 10000  # max is 250,000 samples per second >:D
        self.num = int(np.ceil(rate * self.t))  # number of samples to measure
        # times = np.arange(start=0, stop=self.t, step=(1 / rate))

        # setting up frequency variables
        min_freq = 1 / self.t
        max_freq = rate / 2  # = (num / 2) / t  # aka Nyquist frequency
        # self.num_freqs = (self.max_freq - 0) / self.min_freq
        self.freqs = np.linspace(start=min_freq, stop=max_freq, num=int((self.num / 2) - 1), endpoint=True)

        # setting up lab equipment
        self.sig_gen = set_up_signal_generator_pulse()
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(dev_signal, min_val=-10.0, max_val=10.0)
        self.task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=self.num)
        self.task.ai_channels.add_ai_voltage_chan(dev_temp, min_val=-10.0, max_val=10.0)  # todo multiple reads??

    def auto_temp_pulse(self, temp_start=30, temp_stop=60, temp_step=5, temp_repeats=3, cutoff=None):
        if cutoff is None:
            cutoff = [0.25, 0.75]
        required_temps = np.repeat(
            np.linspace(start=temp_start, stop=temp_stop, num=int(1 + abs(temp_stop - temp_start) / temp_step)),
            repeats=temp_repeats)
        data_list = np.ones([len(self.freqs), len(required_temps)]) * np.nan
        temps = np.ones([len(required_temps)]) * np.nan
        overall_start = time.time()
        with open("outputs/autotemp.txt", "w") as autotemp:
            for i, temp_should_be in enumerate(required_temps):
                temp = np.nanmean(temp_get(self.task.read(self.num)[1]))
                while (temp_start < temp_stop and temp < temp_should_be - 0.5)\
                        or (temp_start > temp_stop and temp > temp_should_be + 0.5):
                    # repeatedly check if the temp is high/low enough to move on
                    print(f"temp is {temp:.3g}")
                    time.sleep(10)
                    temp = np.nanmean(temp_get(self.task.read(self.num)[1]))
                data, temp = self.measure()
                data_list[:, i] = data
                temps[i] = float(temp)

                # file management
                arr = np.zeros([len(data), 2])
                arr[:, 0] = self.freqs
                arr[:, 1] = data
                np.savetxt("outputs/output.txt", arr)
                resave_output(method="TP", save_path=self.save_folder_path, temperature=f"{temp:.3g}",
                              sample_name=self.sample_name)

                # fit
                freqs = self.freqs[int(len(data) * cutoff[0]):int(len(data) * cutoff[1])]
                data = data[int(len(data) * cutoff[0]):int(len(data) * cutoff[1])]
                out = curve_fit(f=lorentzian, xdata=freqs, ydata=data, bounds=([0, 50, 0, 0], [1e4, 5000, 2, 1e4]))
                value, error = out[0], out[1]
                error = np.sqrt(np.diag(error))
                # peaks[i] = value[1]
                # errors[i] = error[1]
                print(f"Temp {temp:.3g}, peak at {value[1]:.6g} pm {error[1]:.2g} Hz")
                autotemp.write(f"{temp:.6g} {value[1]:.6g} {error[1]:.6g}\n")
        # arr = np.zeros([len(temps), 3])
        # arr[:, 0] = temps
        # arr[:, 1] = peaks
        # arr[:, 2] = errors
        # np.savetxt("outputs/autotemp.txt", arr, fmt='%.6g')
        print(f"That took {time.time() - overall_start:.6g} seconds")
        resave_auto(save_path=self.save_folder_path, sample_name=self.sample_name, method="P")
        # plt.errorbar(temps, peaks, yerr=errors, fmt='x')
        # plt.xlabel("Temperature / Celsius")
        # plt.ylabel("Peak/ Hz")
        # plt.show()  # todo editing

    def measure(self):  # returns response to each frequency and the average temperature during the measurement
        self.sig_gen.write("OUTPut ON")
        data_list = np.ones([len(self.freqs), self.runs]) * np.nan
        # temps = np.ones([self.runs]) * np.nan

        for i in range(self.runs + 1):
            complete = False
            while not complete:
                # reset signal generator output to get to a known timing
                self.sig_gen.write(f'APPLy:PULSe {10}, MAX')
                self.sig_gen.write(f'APPLy:PULSe {1 / self.t}, MAX')
                start = time.time()
                if i > 0:  # do processing of previous signal
                    # discount the pulse and everything before it
                    response = np.where(range(len(signal)) > np.argmax(np.abs(signal)) + 10, signal, 0)
                    # process and store
                    data = np.abs(np.fft.fft(response - np.mean(response))[1:int(self.num / 2)])
                    data_list[:, i - 1] = data
                if self.sleep_time - (time.time() - start) > 1e-3:
                    time.sleep(self.sleep_time - (time.time() - start))  # wait for next cycle
                else:
                    # wait for the cycle after the next cycle
                    self.sig_gen.write(f'APPLy:PULSe {10}, MAX')
                    self.sig_gen.write(f'APPLy:PULSe {1 / self.t}, MAX')
                    time.sleep(self.sleep_time)
                # read the raw microphone signal & temperature voltages
                signal, temps = self.task.read(self.num)
                if np.argmax(np.abs(signal)) < len(signal) / 3:
                    complete = True
        self.sig_gen.write("OUTPut OFF")
        return np.nanmean(data_list, 1), np.nanmean(temp_get(temps))

    def close(self):
        self.task.close()
        self.sig_gen.write("OUTPut OFF")
        self.sig_gen.close()

    def calibrate(self):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([0, 0], [0, 0])
        data = []
        temps = []
        while True:
            d, t = self.task.read(self.num)
            data.append(np.mean(d))
            temps.append(np.mean(t))
            plt.title(f"temp voltage is {np.mean(t):.4g}")
            # print(np.mean(t))
            line.set_xdata(range(len(temps)))
            line.set_ydata(temps)
            ax.set_xlim([0, len(temps)])
            ax.set_ylim(ax_lims(temps))
            fig.canvas.draw()
            fig.canvas.flush_events()
