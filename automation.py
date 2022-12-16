import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
# from tqdm import tqdm
import time
from scipy.optimize import curve_fit
from noisyopt import minimizeCompass

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
    return (100 / (hundred - zero)) * (np.asarray(voltage) - zero)


def convert_temp_to_tempstr(temp):
    # works for any temp from 10.00 to 99.99, not sure about any others though!
    tempstr = f"{temp:.4g}"
    if len(tempstr.split(".")) == 1:
        return tempstr + ".00"
    elif len(tempstr) < 5:
        return tempstr + "0"
    return tempstr


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
        self.runs = int(33)
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

        # storage for adaptive
        self.out = open("outputs/output_a.txt", 'w')

        # setting up lab equipment
        self.sig_gen = set_up_signal_generator_pulse()
        self.sig_gen.write("OUTPut OFF")
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(dev_signal, min_val=-10.0, max_val=10.0)
        self.task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=self.num)
        self.task.ai_channels.add_ai_voltage_chan(dev_temp, min_val=-10.0, max_val=10.0)  # todo multiple reads??z

    def auto_temp_pulse(self, cutoff=None, **kwargs):
        if cutoff is None:
            cutoff = [0.25, 0.75]
        required_temps, up = self.required_temps_get(**kwargs)
        data_list = np.ones([len(self.freqs), len(required_temps)]) * np.nan
        temps = np.ones([len(required_temps)]) * np.nan
        overall_start = time.time()
        with open("outputs/autotemp.txt", "w") as autotemp:
            for i, temp_should_be in enumerate(required_temps):
                temp = self.temp_move_on(temp_should_be, up)

                data, temp = self.measure_pulse()
                data_list[:, i] = data
                temps[i] = float(temp)

                # file management
                arr = np.zeros([len(data), 2])
                arr[:, 0] = self.freqs
                arr[:, 1] = data
                np.savetxt("outputs/output.txt", arr)
                resave_output(
                    method=f"TP{str(i).zfill(len(required_temps))}", save_path=self.save_folder_path + r"\Auto",
                    temperature=convert_temp_to_tempstr(temp), sample_name=self.sample_name)

                # fit
                freqs = self.freqs[int(len(data) * cutoff[0]):int(len(data) * cutoff[1])]
                data = data[int(len(data) * cutoff[0]):int(len(data) * cutoff[1])]
                out = curve_fit(f=lorentzian, xdata=freqs, ydata=data, bounds=([0, 50, 0, 0], [1e4, 5000, 2, 1e4]))
                value, error = out[0], out[1]
                error = np.sqrt(np.diag(error))
                print(f"Temp {temp:.3g}, peak at {value[1]:.6g} pm {error[1]:.2g} Hz")
                autotemp.write(f"{temp:.6g} {value[1]:.6g} {error[1]:.6g}\n")
        print(f"That took {time.time() - overall_start:.6g} seconds")
        resave_auto(save_path=self.save_folder_path, sample_name=self.sample_name, method="P")

    def auto_temp_sweep(self, freq=None, freqstep=5, **kwargs):
        if freq is None:
            freq = [50, 5000]
        required_temps, up = self.required_temps_get(**kwargs)
        freqs = np.sort(np.linspace(start=freq[0], stop=freq[-1], num=int(1 + abs(freq[-1] - freq[0]) / freqstep)))
        data_list = np.ones([len(freqs), len(required_temps)]) * np.nan
        # temps = np.ones([len(required_temps)]) * np.nan

        overall_start = time.time()
        with open("outputs/autotemp.txt", "w") as autotemp:
            for i, temp_should_be in enumerate(required_temps):
                temp = self.temp_move_on(temp_should_be, up)

                data, temp = self.measure_sweep(freqs)
                data_list[:, i] = data
                # temps[i] = float(temp)

                # file management
                arr = np.zeros([len(data), 2])
                arr[:, 0] = freqs
                arr[:, 1] = data
                np.savetxt("outputs/output.txt", arr)
                resave_output(
                    method=f"TS{str(i).zfill(len(required_temps))}", save_path=self.save_folder_path + r"\Auto",
                    temperature=convert_temp_to_tempstr(temp), sample_name=self.sample_name)

                # fit
                out = curve_fit(f=lorentzian, xdata=freqs, ydata=data, bounds=([0, freq[0], 0, 0],
                                                                               [1e4, freq[-1], 2, 1e4]))
                value, error = out[0], out[1]
                error = np.sqrt(np.diag(error))
                print(f"Temp {temp:.3g}, peak at {value[1]:.6g} pm {error[1]:.2g} Hz")
                autotemp.write(f"{temp:.6g} {value[1]:.6g} {error[1]:.6g}\n")

        print(f"That took {time.time() - overall_start:.4g} seconds")
        resave_auto(save_path=self.save_folder_path, sample_name=self.sample_name, method="S")

    def auto_temp_adaptive(self, tolerance=5, start_guess=1e3, deltainit=1e3, bounds=None, **kwargs):
        if bounds is None:
            bounds = [100, 4e3]
        required_temps, up = self.required_temps_get(**kwargs)

        overall_start = time.time()
        with open("outputs/autotemp.txt", "w") as autotemp:
            for i, temp_should_be in enumerate(required_temps):
                temp = self.temp_move_on(temp_should_be, up)

                res = None
                while res is None:
                    try:
                        res = minimizeCompass(
                            self.measure_adaptive, x0=[start_guess], bounds=[bounds], errorcontrol=True, disp=False,
                            paired=False, deltainit=deltainit, deltatol=tolerance, funcNinit=4, funcmultfactor=1.25)
                    except ExitException:
                        continue
                # this is the best way I can think to stop minimizeCompass going for too long without reaching into it

                # file management
                self.out.close()
                np.savetxt("outputs/output.txt", np.loadtxt("outputs/output_a.txt"), fmt='%.6g')
                self.out = open("outputs/output_a.txt", "w")
                resave_output(
                    method=f"TA{str(i).zfill(len(required_temps))}", save_path=self.save_folder_path + r"\Auto",
                    temperature=convert_temp_to_tempstr(temp), sample_name=self.sample_name)

                print(f"Temp {temp:.3g}, peak at {res.x[0]:.6g} pm {tolerance /2:.2g} Hz")
                autotemp.write(f"{temp:.6g} {res.x[0]:.6g} {tolerance / 2:.6g}\n")
        print(f"That took {time.time() - overall_start:.4g} seconds")
        resave_auto(save_path=self.save_folder_path, sample_name=self.sample_name, method="A")

    def measure_pulse(self):  # returns response to each frequency and the average temperature during the measurement
        self.sig_gen.write("OUTPut ON")
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
                    response = np.where(range(len(signal)) > np.argmax(np.abs(signal)) + 30, signal, 0)
                    # process and store
                    data = np.abs(np.fft.fft(response - np.mean(response))[1:int(self.num / 2)])
                    data_list[:, i - 1] = data
                    temps[i - 1] = np.mean(temp)
                if self.sleep_time - (time.time() - start) > 1e-3:
                    time.sleep(self.sleep_time - (time.time() - start))  # wait for next cycle
                else:
                    # wait for the cycle after the next cycle
                    self.sig_gen.write(f'APPLy:PULSe {10}, MAX')
                    self.sig_gen.write(f'APPLy:PULSe {1 / self.t}, MAX')
                    time.sleep(self.sleep_time)
                # read the raw microphone signal & temperature voltages
                signal, temp = self.task.read(self.num)
                if np.argmax(np.abs(signal)) < len(signal) / 3:
                    complete = True
        self.sig_gen.write("OUTPut OFF")
        return np.nanmean(data_list, 1), temp_get(np.nanmean(temps))

    def measure_sweep(self, freqs):  # returns response to each frequency and the average temperature during the measurement
        self.sig_gen.write("OUTPut ON")
        data = np.ones([len(freqs)]) * np.nan
        temps = np.ones([len(freqs)]) * np.nan

        a = np.arange(len(freqs))
        np.random.default_rng().shuffle(a)
        for e in a:
            self.sig_gen.write(f'APPLy:SINusoid {freqs[e]}, 10')  # todo 10V or 5V?
            time.sleep(0.05)
            signal, temp = self.task.read(self.num)
            data[e] = np.sqrt(np.mean(np.square(signal)))  # calculate RMS of signal
            temps[e] = np.mean(temp)

        self.sig_gen.write("OUTPut OFF")
        return data, temp_get(np.nanmean(temps))

    def measure_adaptive(self, f):
        self.sig_gen.write(f'APPLy:SINusoid {float(f)}, 10')  # set the signal generator to the desired frequency
        time.sleep(0.05)
        signal, temps = self.task.read(self.num)
        rms = np.sqrt(np.mean(np.square(signal)))  # read signal from microphone then calculate RMS
        self.out.write(f"{float(f):.6g} {rms:.6g} {temp_get(np.mean(temps)):.6g}\n")
        if np.random.default_rng().random() > 1 - 1 / 400:  # 1 in 400 times
            self.out.close()
            with open("outputs/output_a.txt", 'r') as f:
                file_length = len(f.readlines())
            # print(f"minimizeCompass has performed {file_length} measurements")
            if file_length > 200:  # length greater than 200 measurements
                print("minimizeCompass got stuck so this measurement is being restarted")
                self.out = open("outputs/output_a.txt", 'w')
                raise ExitException
            else:
                self.out = open("outputs/output_a.txt", 'a')
        return -rms

    def close(self):
        self.task.close()
        self.sig_gen.write("OUTPut OFF")
        self.sig_gen.close()
        self.out.close()

    def calibrate(self):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([0, 0], [0, 0], 'k')
        axt = ax.twinx()
        linet, = axt.plot([0, 0], [0, 0], 'r')
        data = []
        temps = []
        realts = []
        while True:
            d, t = self.task.read(self.num)
            data.append(np.mean(d))
            temps.append(np.mean(t))
            realt = np.mean(temp_get(t))
            realts.append(realt)
            plt.title(f"temp voltage is {np.mean(t):.4g}, meaning temp is {realt:.4g} C")
            # print(np.mean(t))

            line.set_xdata(range(len(temps)))
            line.set_ydata(temps)
            ax.set_xlim([0, len(temps)])
            ax.set_ylim(ax_lims(temps))

            linet.set_xdata(range(len(realts)))
            linet.set_ydata(realts)
            axt.set_xlim([0, len(realts)])
            axt.set_ylim(ax_lims(realts))

            fig.canvas.draw()
            fig.canvas.flush_events()

    def required_temps_get(self, temp_start=None, temp_stop=None, temp_step=2.5, temp_repeats=1):
        if temp_start is None:
            temp_start = input("start temp ('C' sets as current):")
            if temp_start == "C":
                temp_start = temp_get(np.nanmean(self.task.read(self.num)[1]))
        temp_start = float(temp_start)
        print(f"Current temp is {temp_start:.4g}")
        if temp_stop is None:
            temp_stop = float(input("stop temp:"))

        # getting correct starting point to make intervals nice
        if not np.isclose(temp_start % temp_step, 0):
            if temp_start < temp_stop:
                temp_start = temp_start - temp_start % temp_step
            else:
                temp_start = temp_start + (temp_step - temp_start % temp_step)
        print(f"start temp is {temp_start:.4g}")

        # if temp_start == temp_stop:
        #     raise ValueError("Don't use AutoTemp for single measurements")

        return np.repeat(
            np.linspace(start=temp_start, stop=temp_stop, num=int(1 + abs(temp_stop - temp_start) / temp_step)),
            repeats=temp_repeats), temp_start < temp_stop

    def temp_move_on(self, temp_should_be, up):
        # only move on to the next measurements after the temperature has been reached
        temp = np.nanmean(temp_get(self.task.read(self.num)[1]))
        while (up and temp < temp_should_be - 0.25) or (not up and temp > temp_should_be + 0.25):
            # repeatedly check if the temp is high/low enough to move on (if it is not enough it will stay here)
            # print(f"temp is {convert_temp_to_tempstr(temp)}")
            time.sleep(1)
            temp = np.nanmean(temp_get(self.task.read(self.num)[1]))
        return temp


class ExitException(Exception):
    pass
