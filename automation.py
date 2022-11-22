import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from tqdm import tqdm
import time

from my_tools import resave_output
from fitting import fit_fast
from IO_setup import set_up_signal_generator_sine, set_up_signal_generator_pulse


class AutoMeasure:
    """
    Automatically measures at given temperatures

    input 'task' is:
    with nidaqmx.Task() as task:
        m = AutoMeasure("outputs", "C10", task)
    """

    def __init__(self, save_folder_path, sample_name, task):
        self.save_folder_path = save_folder_path
        self.sample_name = sample_name

        # setting up data collection variables
        self.runs = int(10)
        self.t = 0.2
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
        self.task = task
        self.task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10.0, max_val=10.0)
        self.task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=self.num)

    def auto_temp_pulse(self, temp_step=1, temp_end=60):
        temp_start = self.temp_get()
        temps = np.linspace(temp_start, temp_end, num=int((temp_end - temp_start) / temp_step))
        peaks = np.ones_like(temps) * np.nan
        for i, temp in enumerate(temps):
            while self.temp_get() < temps[i]:
                time.sleep(1)
            print(self.temp_get())
            data_list = self.measure(temp)  # todo temp needs to be in a filename friendly format!
            peaks[i] = fit_fast(self.freqs, data_list)[1][1]
        plt.plot(temps, peaks, 'x')

    def temp_get(self):
        # todo average over a short time!
        temp = self.task  # todo get current temp somehow please and thanks
        return float(temp)

    def measure(self, temp):
        data_list = np.ones([len(self.freqs), self.runs]) * np.nan

        for i in tqdm(range(self.runs)):
            # todo do 1 pulse then measure, so we don't have to see the pulse, just the remaining resonance.
            p = 1 / (self.t * (0.1 + 0.5 * np.random.random()))  # randomise signal duration
            self.sig_gen.write(f'APPLy:PULSe {p}, MAX')  # set signal duration
            signal = self.task.read(self.num)  # read the raw microphone signal
            # process
            data_list[:, i] = np.abs(np.fft.fft(signal - np.mean(signal))[1:int(self.num / 2)])

        # file management
        np.savetxt("outputs/output.txt", np.nanmean(data_list, 1))
        resave_output(method="AP", temperature=temp, sample_name=self.sample_name)
        return np.nanmean(data_list, 1)
