import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
# from tqdm import tqdm
import time
from scipy.optimize import curve_fit

from my_tools import resave_output
from fitting import lorentzian
from IO_setup import set_up_signal_generator_pulse  # , set_up_signal_generator_sine
# from measurement import measure, measure_adaptive, measure_pulse_decay


def temp_get(temp):  # processes an array of voltages to return the corresponding array of temps (can be len = 1)
    temp = np.asarray(temp)
    temp = temp * 100 - 5  # todo calibrate
    return temp


class AutoMeasure:
    """
    Automatically measures at given temperatures

    input 'task' is:
    with nidaqmx.Task() as task:
        m = AutoMeasure("outputs", "C10", task)
    """

    def __init__(self, save_folder_path, sample_name, dev_signal, dev_temp):
        self.save_folder_path = save_folder_path
        self.sample_name = sample_name

        # setting up data collection variables
        self.runs = int(100)
        self.t = 0.2
        self.sleep_time = 0.135
        rate = 10001  # max is 250,000 samples per second >:D
        self.num = int(np.ceil(rate * self.t))  # number of samples to measure
        times = np.arange(start=0, stop=self.t, step=(1 / rate))

        # setting up frequency variables
        min_freq = 1 / self.t
        max_freq = rate / 2  # = (num / 2) / t  # aka Nyquist frequency
        # self.num_freqs = (self.max_freq - 0) / self.min_freq
        self.freqs = np.linspace(start=min_freq, stop=max_freq, num=int((self.num / 2) - 1), endpoint=True)
        # todo this shouldn't be the wrong size!!!!!!!!!!!!!! int((num / 2) - 1) or int((num - 1) / 2)),

        # setting up lab equipment
        self.sig_gen = set_up_signal_generator_pulse()
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(dev_signal, min_val=-10.0, max_val=10.0)
        self.task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=self.num)
        self.task.ai_channels.add_ai_voltage_chan(dev_temp, min_val=-10.0, max_val=10.0)  # todo multiple reads??

    def auto_temp_pulse(self, time_between_measurement_starts=60, total_measurements=60):
        data_list = np.ones([len(self.freqs), total_measurements]) * np.nan
        temps, peaks, errors = np.ones([total_measurements]) * np.nan
        overall_start = time.time()
        for i in range(total_measurements):
            while time_between_measurement_starts * i + overall_start < time.time():
                time.sleep(0.1)
            data, temp = self.measure()
            data_list[:, i] = data
            temps[i] = temp

            # file management
            np.savetxt("outputs/output.txt", data)
            resave_output(method="AP", save_path=self.save_folder_path, temperature=str(temp),  # todo str(temp)
                          sample_name=self.sample_name)

            # fit
            out = curve_fit(f=lorentzian, xdata=self.freqs, ydata=data, bounds=([0, 50, 0, 0], [1e4, 5000, 2, 1e4]))
            peaks[i] = out[0][1]
            errors[i] = out[1][1]
        arr = np.zeros([len(temps), 3])
        arr[:, 0] = temps
        arr[:, 1] = peaks
        arr[:, 2] = errors
        np.savetxt("outputs/auto.txt", arr)
        plt.errorbar(temps, peaks, yerr=errors, fmt='x')
        plt.show()  # todo editing

    def measure(self):  # returns response to each frequency and the average temperature during the measurement
        data_list = np.ones([len(self.freqs), self.runs]) * np.nan
        temps = np.ones([self.runs]) * np.nan

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
                signal, temps[i] = self.task.read(self.num)
                if np.argmax(np.abs(signal)) < len(signal) / 3:
                    complete = True
        return np.nanmean(data_list, 1), np.nanmean(temp_get(temps))

    def close(self):
        self.task.close()
        self.sig_gen.write("OUTPut OFF")
        self.sig_gen.close()
