import io
import numpy as np
import matplotlib.pyplot as plt
import nidaqmx as ni
# from tqdm import tqdm
import time
from scipy.optimize import curve_fit
from noisyopt import minimizeCompass

from my_tools import resave_output, resave_auto, ax_lims, temp_get, convert_temp_to_tempstr
from fitting import lorentzian
from IO_setup import set_up_signal_generator_pulse  # , set_up_signal_generator_sine
# from measurement import measure, measure_adaptive, measure_pulse_decay
from aggregation import resave


def grab_temp(dev, chan, num=1000):
    with ni.Task() as task:
        task.ai_channels.add_ai_voltage_chan(dev + '/' + chan, min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=10000, samps_per_chan=10000)
        return np.mean(temp_get(task.read(num)))


def set_up_daq(mode, c1, c2, rate=int(20e3), t=0.2):
    task = ni.Task()
    match mode:
        case 'dual':
            num = int(np.ceil(int(rate / 2) * t))  # number of samples to measure
            task.ai_channels.add_ai_voltage_chan(c1, min_val=-10.0, max_val=10.0)
            task.timing.cfg_samp_clk_timing(rate=int(rate / 2), samps_per_chan=num)
            task.ai_channels.add_ai_voltage_chan(c2, min_val=-10.0, max_val=10.0)
        case 'single':
            num = int(np.ceil(rate * t))  # number of samples to measure
            task.ai_channels.add_ai_voltage_chan(c1, min_val=-10.0, max_val=10.0)
            task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        case _:
            raise ValueError("'dual' or 'single' mode for set_up_daq_task")
    return task, num


def fitstuff(data, freqs, bounds, temp, overall_start, freqstep=5):
    # fit
    # (cursed function) (why is this even here for autotemp? mayb i just remove it.)
    # todo remove this function at some point plz
    try:
        out = curve_fit(f=lorentzian, xdata=freqs, ydata=data, bounds=([0, bounds[0], 0, 0],
                                                                       [1e5, bounds[-1], 2, 1e5]))
        value, error = out[0], out[1]
        error = np.sqrt(np.diag(error))
        if error[1] > 2e1:  # todo tweak
            # print("curve_fit returned a value with too low confidence")
            raise RuntimeError
        print(f"\rTemp {temp:.3g}, peak at {value[1]:.6g} pm {error[1]:.2g} Hz", end='')
    except RuntimeError:
        value = [0, freqs[np.argmax(data)]]
        error = [0, freqstep * 2]
        # print(f"The curve_fit failed. Using fmax: {value[1]:.6g} Hz and error will be {error[1]:.2g} Hz")
        print(f"\rTemp {temp:.3g}, peak at {value[1]:.6g} pm {error[1]:.2g} Hz - curve_fit failed", end='')
    with open("outputs/autotemp.txt", "a") as autotemp:
        autotemp.write(f"{time.time() - overall_start:.6g}  {value[1]:.6g} {error[1]:.6g} {temp:.6g}\n")


def finishstuff(overall_start, save_folder_path, sample_name, method):
    total_t = time.time() - overall_start
    print(f"\rThat took {total_t:.6g} seconds")
    print(f"Or {total_t // (60 * 60)}h {(total_t % (60 * 60)) // 60}m {total_t % 60:.4g}s")
    resave_auto(save_path=save_folder_path, sample_name=sample_name, method=method)


class Measurer:
    """
    AutoTemp calls this to do measurements
    """

    def __init__(self, out, task, num, sig_gen, vpp):
        self.out = out
        self.task = task
        self.num = num
        self.sig_gen = sig_gen
        self.vpp = vpp

    def measure_adaptive(self, f=None, i=[0]):
        if f is None:
            a = i[0]
            i[0] = 0  # reset counter
            return a
        self.sig_gen.write(f'APPLy:SINusoid {float(f)}, {self.vpp}')  # set the signal generator to the frequency f
        time.sleep(0.05)
        signal, temps = self.task.read(self.num)
        rms = np.sqrt(np.mean(np.square(signal)))  # read signal from microphone then calculate RMS
        self.out.write(f"{float(f):.6g} {rms:.6g} {temp_get(np.mean(temps)):.6g}\n")
        if i[0] >= 2e2:
            # credit for this neat mutable argument call counter trick goes to https://stackoverflow.com/a/23160861
            print("minimizeCompass got stuck so is being restarted")
            self.out.close()
            self.out = open("outputs/output_a.txt", 'w')  # reopen (and empty) the output file
            i[0] = 0
            raise ExitException
        i[0] += 1
        return -rms

    def measure_pulse(self, freqs, sleep_time, num, runs=33):
        # returns response to each frequency and the average temperature during the measurement
        runs = int(runs)

        self.sig_gen.write("OUTPut ON")
        data_list = np.ones([len(freqs), runs]) * np.nan

        for i in range(runs + 1):
            complete = False
            while not complete:
                # reset signal generator output to get to a known timing
                self.sig_gen.write(f'APPLy:PULSe {10}, MAX')
                self.sig_gen.write(f'APPLy:PULSe {1 / self.t}, MAX')
                start = time.time()
                if i > 0:  # do processing of previous signal
                    response = signal
                    # process and store
                    data_list[:, i - 1] = np.abs(np.fft.fft(response - np.mean(response))[1:int(num / 2) + 1])
                # todo look into the timings... I can surely speed this up?
                if sleep_time - (time.time() - start) > 1e-3:
                    time.sleep(sleep_time - (time.time() - start))  # wait for next cycle
                else:
                    # wait for the cycle after the next cycle
                    self.sig_gen.write(f'APPLy:PULSe {10}, MAX')
                    self.sig_gen.write(f'APPLy:PULSe {1 / self.t}, MAX')
                    time.sleep(sleep_time)
                # read the microphone signal
                signal = self.task.read(num)
                if np.argmax(np.abs(signal)) < len(signal) / 3:
                    complete = True
        self.sig_gen.write("OUTPut OFF")
        return np.nanmean(data_list, 1)

    def measure_sweep(self, freqs):  # returns response to each frequency and the average temperature during measurement
        self.sig_gen.write("OUTPut ON")
        data = np.ones([len(freqs)]) * np.nan
        temps = np.ones([len(freqs)]) * np.nan

        a = np.arange(len(freqs))
        np.random.default_rng().shuffle(a)
        for e in a:
            self.sig_gen.write(f'APPLy:SINusoid {freqs[e]}, {self.vpp}')
            time.sleep(0.05)
            signal, temp = self.task.read(self.num)
            data[e] = np.sqrt(np.mean(np.square(signal)))  # calculate RMS of signal
            temps[e] = np.mean(temp)

        self.sig_gen.write("OUTPut OFF")
        return data, temp_get(np.nanmean(temps))


class AutoTemp:
    """
    Automatically measures at given temperatures

    files will be saved for each temperature that is measured, and a final file that saves all the peaks and their temps

    """

    def __init__(self, save_folder_path, dev_signal, dev_temp, vpp=10, sample_name=None, bounds=None, t=None):
        self.save_folder_path = save_folder_path
        if sample_name is None:
            self.sample_name = input("Sample name:")
        else:
            self.sample_name = sample_name
        if bounds is None:
            self.bounds = [float(input("lower freq:")), float(input("upper freq:"))]
        else:
            self.bounds = bounds
        if t is None:
            t = 0.2

        self.dev_signal = dev_signal
        self.dev_temp = dev_temp

        # setting up data collection
        self.rate = 20000  # max samples per second of the DAQ card
        self.t = t
        self.task, self.num = set_up_daq(mode='dual', c1=self.dev_signal, c2=self.dev_temp, rate=self.rate, t=self.t)

        # storage for adaptive
        self.out = open("outputs/output_a.txt", 'w')

        # setting up signal generator
        self.sig_gen = set_up_signal_generator_pulse()
        self.sig_gen.write("OUTPut OFF")

        self.M = Measurer(self.out, self.task, self.num, self.sig_gen, vpp)

    def auto_pulse(self, bounds=None, time_between=30, repeats=300, runs=33, temp=None, **kwargs):
        if temp is None:
            temp = input("Do you want temperature recordings? 'Y' for yes:")
        if temp == 'y' or temp == 'Y':
            temp = True
        else:
            temp = 0
        if bounds is None:
            bounds = [1e3, 4e3]
        cutoff = np.divide(bounds, 10e3)
        repeats = int(repeats)
        if repeats < 2:
            repeats = int(2)

        sleep_time = 0.135
        # times = np.arange(start=0, stop=self.t, step=(1 / rate))

        # setting up frequency variables
        num = int(np.ceil(self.rate * self.t))  # number of samples to measure
        min_freq = 1 / self.t
        max_freq = self.rate / 2  # = (num / 2) / t  # aka Nyquist frequency
        # self.num_freqs = (self.max_freq - 0) / self.min_freq
        freqs = np.linspace(start=min_freq, stop=max_freq, num=int(num / 2), endpoint=True)
        data_list = np.ones([len(freqs), repeats]) * np.nan

        print("starting autotemp...")
        self.task.close()
        self.task, num = set_up_daq(mode='single', c1=self.dev_signal, c2=self.dev_temp, rate=self.rate, t=self.t)
        with open("outputs/autotemp.txt", "w") as autotemp:  # reset the file
            pass

        overall_start = time.time()
        for i in range(repeats):
            print(f"\r {i} of {repeats}", end='')
            data = self.M.measure_pulse(freqs=freqs, sleep_time=sleep_time, num=num, runs=runs)

            # data/file management
            data_list[:, i] = data
            arr = np.zeros([len(data), 2])
            arr[:, 0] = freqs
            arr[:, 1] = data
            np.savetxt("outputs/output.txt", arr)
            resave_output(method=f"TP{str(i).zfill(len(str(repeats - 1)))}",
                          save_path=self.save_folder_path + r"\Spectra", temperature="T", sample=self.sample_name)

            fitstuff(data, freqs, bounds, temp, overall_start)

            sleep = time_between * (i + 1) - (time.time() - overall_start)
            if sleep > 0:
                time.sleep(sleep)

        finishstuff(overall_start, self.save_folder_path, self.sample_name, method="P")
        fname = '_'.join([str(e).zfill(2) for e in time.localtime()[0:5]]) + f"_TP_{self.sample_name}.txt"
        # np.savetxt(self.save_folder_path + "/" + fname, np.loadtxt(f"outputs/autotemp.txt"))
        resave(self.save_folder_path, name=fname)

    def auto_temp_pulse(self, delay=20, time_between=30, runs=100, **kwargs):
        bounds = self.bounds
        cutoff = np.divide(bounds, 10e3)

        sleep_time = 0.135
        # times = np.arange(start=0, stop=self.t, step=(1 / rate))

        # setting up frequency variables
        num = int(np.ceil(self.rate * self.t))  # number of samples to measure
        min_freq = 1 / self.t
        max_freq = self.rate / 2  # = (num / 2) / t  # aka Nyquist frequency
        # self.num_freqs = (self.max_freq - 0) / self.min_freq
        freqs = np.linspace(start=min_freq, stop=max_freq, num=int((num / 2) - 1), endpoint=True)

        print("starting autotemp...")

        self.task.close()
        self.task = set_up_daq(mode='dual', c1=self.dev_signal, c2=self.dev_temp, rate=self.rate, t=self.t)[0]
        required_temps, up = self.required_temps_get(**kwargs)
        data_list = np.ones([len(freqs), len(required_temps)]) * np.nan
        temps = np.ones([len(required_temps)]) * np.nan
        overall_start = time.time()
        with open("outputs/autotemp.txt", "w") as autotemp:  # reset the file
            pass

        for i, temp_should_be in enumerate(required_temps):
            print(f"\r {i} of {len(required_temps)}", end='')
            temp = self.temp_move_on(temp_should_be, up)

            self.task.close()
            self.task, num = set_up_daq(mode='single', c1=self.dev_signal, c2=self.dev_temp, rate=self.rate, t=self.t)
            data = self.M.measure_pulse(freqs=freqs, sleep_time=sleep_time, num=num, runs=runs)
            self.task.close()
            self.task = set_up_daq(mode='dual', c1=self.dev_signal, c2=self.dev_temp, rate=self.rate, t=self.t)[0]
            temps[i] = (float(temp) + float(np.nanmean(temp_get(self.task.read(self.num)[1])))) / 2

            # data/file management
            data_list[:, i] = data
            arr = np.zeros([len(data), 2])
            arr[:, 0] = freqs
            arr[:, 1] = data
            np.savetxt("outputs/output.txt", arr)
            resave_output(method=f"TP{str(i).zfill(len(str(len(required_temps))))}",
                          save_path=self.save_folder_path + r"\Spectra", temperature=convert_temp_to_tempstr(temp),
                          sample=self.sample_name)

            fitstuff(data, freqs, bounds, temp, overall_start)

        finishstuff(overall_start, self.save_folder_path, self.sample_name, method="P")

    def auto_sweep(self, freqstep=5, repeats=None, temp=None):
        if temp is None:
            temp = input("Do you want temperature recordings? 'Y' for yes:")
        if temp == 'y' or temp == 'Y':
            temp = True
        else:
            temp = 0
        repeats = int(repeats)
        if repeats < 2:
            repeats = int(2)

        print("starting autotemp...")

        bounds = self.bounds
        freqs = np.sort(np.linspace(start=bounds[0], stop=bounds[-1],
                                    num=int(1 + abs(bounds[-1] - bounds[0]) / freqstep)))
        data_list = np.ones([len(freqs), repeats]) * np.nan

        overall_start = time.time()
        with open("outputs/autotemp.txt", "w") as autotemp:  # reset the file
            pass
        for i, temp_should_be in enumerate(range(repeats)):
            print(f"\r {i} of {repeats}", end='')

            data, temp = self.M.measure_sweep(freqs)
            data_list[:, i] = data
            # temps[i] = float(temp)

            # file management
            arr = np.zeros([len(data), 2])
            arr[:, 0] = freqs
            arr[:, 1] = data
            np.savetxt("outputs/output.txt", arr)
            resave_output(method=f"TS{str(i).zfill(len(str(repeats - 1)))}",
                          save_path=self.save_folder_path + r"\Spectra", temperature=convert_temp_to_tempstr(temp),
                          sample=self.sample_name)

            # fitstuff(data, freqs, bounds, temp, overall_start, freqstep)

        finishstuff(overall_start, self.save_folder_path, self.sample_name, method="S")

    def auto_temp_sweep(self, freqstep=5, GUI=None, **kwargs):
        print("starting autotemp...")

        bounds = self.bounds
        required_temps, up = self.required_temps_get(**kwargs)
        freqs = np.sort(np.linspace(start=bounds[0], stop=bounds[-1],
                                    num=int(1 + abs(bounds[-1] - bounds[0]) / freqstep)))
        data_list = np.ones([len(freqs), len(required_temps)]) * np.nan
        # temps = np.ones([len(required_temps)]) * np.nan

        overall_start = time.time()
        with open("outputs/autotemp.txt", "w") as autotemp:  # reset the file
            pass
        for i, temp_should_be in enumerate(required_temps):
            print(f"\r {i} of {len(required_temps)}", end='')
            temp = self.temp_move_on(temp_should_be, up, GUI)

            data, temp = self.M.measure_sweep(freqs)
            data_list[:, i] = data
            # temps[i] = float(temp)

            # file management
            arr = np.zeros([len(data), 2])
            arr[:, 0] = freqs
            arr[:, 1] = data
            np.savetxt("outputs/output.txt", arr)
            resave_output(method=f"TS{str(i).zfill(len(str(len(required_temps) - 1)))}",
                          save_path=self.save_folder_path + r"\Spectra", temperature=convert_temp_to_tempstr(temp),
                          sample=self.sample_name)

            fitstuff(data, freqs, bounds, temp, overall_start, freqstep)

        finishstuff(overall_start, self.save_folder_path, self.sample_name, method="S")

    def auto_temp_adaptive(self, vpp=5, tolerance=5, start_guess=1e3, start_delta=1e3, **kwargs):
        bounds = self.bounds
        if start_guess is None:
            start_guess = 800

        print("starting autotemp...")

        required_temps, up = self.required_temps_get(**kwargs)

        overall_start = time.time()
        with open("outputs/autotemp.txt", "w") as autotemp:  # reset the file
            pass
        for i, temp_should_be in enumerate(required_temps):
            print(f"\r {i} of {len(required_temps)}", end='')
            temp = self.temp_move_on(temp_should_be, up)

            res = None
            while res is None:
                try:
                    res = minimizeCompass(
                        self.M.measure_adaptive, x0=[start_guess], bounds=[bounds], errorcontrol=True, disp=False,
                        paired=False, deltainit=start_delta, deltatol=tolerance, funcNinit=4, funcmultfactor=1.25)
                except ExitException:
                    continue
            # print(f"Optimisation found peak at {res.x[0]} after {self.measure_adaptive()} iterations")
            # this is the best way I can think to stop minimizeCompass going for too long without reaching into it

            # file management
            self.out.close()
            np.savetxt("outputs/output.txt", np.loadtxt("outputs/output_a.txt"), fmt='%.6g')
            self.out = open("outputs/output_a.txt", "w")
            resave_output(method=f"TA{str(i).zfill(len(str(len(required_temps))))}",
                          save_path=self.save_folder_path + r"/Spectra", temperature=convert_temp_to_tempstr(temp),
                          sample=self.sample_name)

            print(f"Temp {temp:.3g}, peak at {res.x[0]:.6g} pm {tolerance /2:.2g} Hz after"
                  f"{self.M.measure_adaptive()} measurements")

            with open("outputs/autotemp.txt", "a") as autotemp:
                autotemp.write(f"{time.time() - overall_start:.6g} {res.x[0]:.6g} {tolerance / 2:.6g} {temp:.6g}\n")
        finishstuff(overall_start, self.save_folder_path, self.sample_name, method="A")

    def close(self):
        self.task.close()
        self.sig_gen.write("OUTPut OFF")
        if (isinstance(self.out, io.TextIOBase) or isinstance(self.out, io.BufferedIOBase) or
                isinstance(self.out, io.RawIOBase) or isinstance(self.out, io.IOBase)):  # if self.out is an open file
            self.out.close()

    def calibrate(self):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([0, 0], [0, 0], 'k')
        axt = ax.twinx()
        linet, = axt.plot([0, 0], [0, 0], 'r')
        # plt.xlabel("elapsed time / s")
        temps = []
        realts = []
        starttime = time.time()
        while plt.fignum_exists(fig.number):
            _, t = self.task.read(self.num)
            temps.append(np.mean(t))
            realt = np.mean(temp_get(t))
            realts.append(realt)
            plt.title(f"reading is {np.mean(t):.6g}V, meaning temp is {realt:.6g}C")
            # print(np.mean(t))

            elapsed = time.time() - starttime
            line.set_xdata(np.linspace(start=0, stop=elapsed, endpoint=True, num=len(temps)))
            line.set_ydata(temps)
            ax.set_xlim([0, elapsed])
            ax.set_ylim(ax_lims(temps))

            linet.set_xdata(np.linspace(start=0, stop=elapsed, endpoint=True, num=len(temps)))
            linet.set_ydata(realts)
            axt.set_xlim([0, elapsed])
            axt.set_ylim(ax_lims(realts))

            fig.canvas.draw()
            fig.canvas.flush_events()

    def required_temps_get(self, temp_start=None, temp_stop=None, temp_step=2.5, temp_repeats=1):
        if temp_start is None:
            temp_start = temp_get(np.nanmean(self.task.read(self.num)[1]))
        temp_start = float(temp_start)
        print(f"Current temp is {temp_start:.4g}")
        if temp_stop is None:
            temp_stop = float(input("stop temp ('NC' to change start temp away from the default current):"))
            if temp_stop == "NC":
                temp_stop = float(input("stop temp:"))
                temp_start = input("start temp:")

        # getting correct starting point to make intervals nice
        if not np.isclose((temp_start - temp_stop) % temp_step, 0):
            if temp_start < temp_stop:
                temp_start = temp_start - temp_start % temp_step
            else:
                temp_start = temp_start + (temp_step - temp_start % temp_step)
        print(f"start temp is {temp_start:.4g}, stop temp is {temp_stop:.4g}, step is {temp_step:.4g}")
        print("starting...", end='')

        # if temp_start == temp_stop:
        #     raise ValueError("Don't use AutoTemp for single measurements")

        return np.repeat(
            np.linspace(start=temp_start, stop=temp_stop, num=int(1 + abs(temp_stop - temp_start) / temp_step)),
            repeats=temp_repeats), temp_start < temp_stop

    def temp_move_on(self, temp_should_be, up, GUI=None):
        # only move on to the next measurements after the temperature has been reached
        temp = np.nanmean(temp_get(self.task.read(self.num)[1]))
        while (up and temp < temp_should_be - 0.25) or (not up and temp > temp_should_be + 0.25):
            # repeatedly check if the temp is high/low enough to move on (if it is not enough it will stay here)
            if GUI is None:
                time.sleep(5)
            else:
                GUI.w.after(5000)  # todo does this work? yes, but it is no better than sleep.

            # printing current temp (trying to avoid annoying formatting)
            if len(f'{temp:.4g}'.split('.')[-1]) == 2 and len(f'{temp:.4g}') == 5:
                temp_str = f'{temp:.4g}'
            elif len(f'{temp:.4g}'.split('.')[-1]) == 1 and len(f'{temp:.4g}') == 4:
                temp_str = f'{temp:.4g}0'
            else:
                temp_str = f'{temp:.4g}'
            print(f"\rMoving on at {temp_should_be + (-0.25 if up else 0.25):.4g}, current temp is {temp_str}", end='')

            temp = np.nanmean(temp_get(self.task.read(self.num)[1]))
        return temp


class ExitException(Exception):
    pass
