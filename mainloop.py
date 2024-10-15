import time
import tkinter as tk
from tkinter.ttk import Combobox
from tkinter import filedialog, scrolledtext
import sys
from matplotlib.pyplot import close as matplotlibclose

from measurement import measure_sweep, measure_pulse_decay, measure_adaptive
from automation import AutoTemp
from IO_setup import list_devices, calibrate, grab_temp
from my_tools import resave_output, round_sig_figs
from aggregation import manual_peak_auto, scatter3d  # , aggregate, colourplot, manual_peak
from fitting import fit  # , find_peaks
from timefrequency import fft_magnitude_and_phase, time_frequency_spectrum2electricboogaloo
from wireplot import wireplot_manager
from audio import play_mic_signal, play_spectra, play_stop, notify_finish


class Main:
    """
    big main class for GUI (which deals with everything else)
    """

    def __init__(self):
        self.w = tk.Tk(className='Piezo actuation and microphone detection')
        self.w.geometry("600x400")  # this is the worst. i hate it.
        self.w.resizable(False, False)

        # set up variables that can be used in buttons
        # self.running = tk.BooleanVar(self.w, value=False)
        self.pause_text = tk.StringVar(self.w, value='Pause')
        self.method = tk.StringVar(self.w, value='P')
        self.run_type = tk.StringVar(self.w, value='single')
        self.repeats = tk.IntVar(self.w, value=1)
        self.counter = tk.IntVar(self.w, value=-1)
        self.fit = tk.BooleanVar(self.w, value=True)
        self.temptrack = tk.BooleanVar(self.w, value=True)
        self.pause = tk.BooleanVar(self.w, value=False)
        self.sample_name = tk.StringVar(self.w, value='SAMPLENAME')
        sp = 'outputs'
        sp = r'C:/Users/mbynmr/Links/datastuff/PSY/Vary temperature'
        self.save_path = tk.StringVar(self.w, value=sp)
        self.t = tk.DoubleVar(self.w, value=0.2)
        self.runs = tk.IntVar(self.w, value=33)
        self.freqstep = tk.DoubleVar(self.w, value=0)
        self.boundU = tk.DoubleVar(self.w, value=4000)
        self.boundL = tk.DoubleVar(self.w, value=50)
        self.tempL = tk.DoubleVar(self.w, value=20)
        self.tempU = tk.DoubleVar(self.w, value=100)
        self.tempstep = tk.DoubleVar(self.w, value=2.5)
        self.temp = tk.DoubleVar(self.w, value=20)
        self.vpp = tk.DoubleVar(self.w, value=10)
        self.dev_signal = tk.StringVar(self.w, value='Dev2')
        self.chan_signal = tk.StringVar(self.w, value='ai0')
        self.dev_temp = tk.StringVar(self.w, value='Dev2')
        self.chan_temp = tk.StringVar(self.w, value='ai1')

        # widgets before mainloop
        self.entry_path = tk.Entry(self.w, textvariable=self.save_path, width=68)
        self.tempbox = tk.Entry(self.w, textvariable=self.temp, width=5)
        self.widgets(*list_devices())

        # print command redirect
        self.print_box = scrolledtext.ScrolledText(self.w, width=52, height=13)
        self.print_box.place(relx=0.01, rely=0.39)

        self.Writer = Writer(self.print_box)

        self.time_start = time.time()
        self.w.after(600, self.update, False)
        self.w.mainloop()

    def widgets(self, devs, chans):
        # some buttons
        tk.Button(self.w, text='Measure', command=self.run).place(relx=0.025, rely=0.075)
        tk.Button(self.w, textvariable=self.pause_text, command=self.pauser).place(relx=0.025, rely=0.175)
        # todo figure out stop functionality
        # tk.Button(self.w, text='Stop', command=self.stop).place(relx=0.2, rely=0.175)
        tk.Button(self.w, text='wavelet', command=buttonfunc).place(relx=0.15, rely=0.075)
        tk.Button(self.w, text='3D plot', command=self.wire).place(relx=0.15, rely=0.175)
        tk.Button(self.w, text='Recover (save) last measure', command=self.resave_output).place(relx=0.12, rely=0.925)
        # tk.Button(self.w, text='Force save auto', command=self.resave_auto).place(relx=0.4, rely=0.925)
        tk.Button(self.w, text='NOTIFY NOISE', command=notify_finish).place(relx=0.4, rely=0.925)
        tk.Button(self.w, text='Manual peaks', command=self.manual_peak).place(relx=0.57, rely=0.925)
        tk.Button(self.w, text='Close', command=self.close).place(relx=0.025, rely=0.925)

        tk.Label(self.w, text="Audio").place(relx=0.85, rely=0.01)
        tk.Button(self.w, text='Play raw signal', command=audio_raw).place(relx=0.85, rely=0.075)
        tk.Button(self.w, text='Play spectra', command=audio_spectra).place(relx=0.85, rely=0.15)
        tk.Button(self.w, text='Stop audio', command=audio_stop).place(relx=0.85, rely=0.225)

        tk.Label(self.w, textvariable=self.counter).place(relx=0.73, rely=0.05)
        tk.Label(self.w, text="Temperature").place(relx=0.625, rely=0.01)
        tk.Entry(self.w, textvariable=self.tempL, width=5).place(relx=0.625, rely=0.19)
        tk.Entry(self.w, textvariable=self.tempU, width=5).place(relx=0.725, rely=0.19)
        tk.Label(self.w, text="< T <").place(relx=0.67, rely=0.19)
        tk.Label(self.w, text="T:").place(relx=0.625, rely=0.09)
        tk.Entry(self.w, textvariable=self.tempstep, width=3).place(relx=0.625, rely=0.25)
        self.tempbox.place(relx=0.65, rely=0.09)
        tk.Checkbutton(self.w, text="track?", variable=self.temptrack).place(relx=0.71, rely=0.09)
        tk.Button(self.w, text='Calibrate', command=self.calibrate).place(relx=0.68, rely=0.25)
        # method options
        tk.Radiobutton(self.w, text='Sweep', variable=self.method, value='S').place(relx=0.25, rely=0.05)
        tk.Radiobutton(self.w, text='Adapt', variable=self.method, value='A').place(relx=0.25, rely=0.125)
        tk.Radiobutton(self.w, text='Pulse', variable=self.method, value='P').place(relx=0.25, rely=0.2)
        tk.Label(self.w, text="Actuation Style").place(relx=0.225, rely=0.01)

        # method options for multiple/single
        tk.Radiobutton(self.w, text='Single run', variable=self.run_type, value='single').place(relx=0.35, rely=0.075)
        tk.Radiobutton(self.w, text='Autotemp', variable=self.run_type, value='autotemp').place(relx=0.35, rely=0.175)
        tk.Label(self.w, text="Repeats?").place(relx=0.38, rely=0.01)

        tk.Label(self.w, text="After").place(relx=0.5, rely=0.01)
        # toggle fit view after
        tk.Checkbutton(self.w, text="Fit after?", variable=self.fit).place(relx=0.48, rely=0.075)
        tk.Entry(self.w, textvariable=self.repeats, width=3).place(relx=0.49, rely=0.2)
        tk.Label(self.w, text="Repeats").place(relx=0.49, rely=0.15)

        # text entries with labels
        tk.Entry(self.w, textvariable=self.sample_name, width=20).place(relx=0.025, rely=0.325)
        tk.Label(self.w, text="Sample name").place(relx=0.025, rely=0.275)
        self.entry_path.place(relx=0.25, rely=0.325)
        self.entry_path.xview_moveto(1)
        tk.Label(self.w, text="Save path").place(relx=0.25, rely=0.275)
        tk.Button(self.w, text='Path', command=self.select_path).place(relx=0.93, rely=0.315)
        x = 0.2  # x shift for lots of things at once
        tk.Entry(self.w, textvariable=self.t, width=5).place(relx=0.7 + x, rely=0.45)
        tk.Label(self.w, text="Measure time").place(relx=0.56 + x, rely=0.45)
        tk.Entry(self.w, textvariable=self.boundL, width=5).place(relx=0.6 + x, rely=0.5)
        tk.Entry(self.w, textvariable=self.boundU, width=5).place(relx=0.7 + x, rely=0.5)
        tk.Label(self.w, text="< f <").place(relx=0.645 + x, rely=0.5)
        tk.Entry(self.w, textvariable=self.runs, width=5).place(relx=0.7 + x, rely=0.6)
        tk.Label(self.w, text="Repeats (pulse)").place(relx=0.555 + x, rely=0.6)  # todo move these to underneath AT
        tk.Entry(self.w, textvariable=self.freqstep, width=5).place(relx=0.7 + x, rely=0.7)
        tk.Label(self.w, text="f step (sweep)").place(relx=0.565 + x, rely=0.7)
        tk.Entry(self.w, textvariable=self.vpp, width=5).place(relx=0.7 + x, rely=0.75)
        tk.Label(self.w, text="Vpp (sweep)").place(relx=0.58 + x, rely=0.75)

        # finally, device & channel dropdowns using Combobox widgets
        Combobox(self.w, values=devs, textvariable=self.dev_signal, width=5).place(relx=0.82, rely=0.85)
        Combobox(self.w, values=chans, textvariable=self.chan_signal, width=5).place(relx=0.91, rely=0.85)
        tk.Label(self.w, text="Signal").place(relx=0.75, rely=0.85)
        Combobox(self.w, values=devs, textvariable=self.dev_temp, width=5).place(relx=0.82, rely=0.9)
        Combobox(self.w, values=chans, textvariable=self.chan_temp, width=5).place(relx=0.91, rely=0.9)
        tk.Label(self.w, text="Temp").place(relx=0.75, rely=0.9)

    def stop(self):
        print("stop")
        print("sry stop doesn't do anything yet")

    def pauser(self):
        self.pause.set(not self.pause.get())
        if self.pause.get():
            self.pause_text.set('Unpause')
        else:
            self.pause_text.set('Pause')

    def update(self, stop=None, i=[0]):
        # credit for this neat mutable argument call counter trick goes to https://stackoverflow.com/a/23160861
        if stop is not None:
            i[0] = 0  # reset counter
            if not stop:
                i[0] = 1
        self.counter.set(i[0])
        if i[0] != 0:
            i[0] += 1
            # try: () except task already using daq card error: (don't continue self.after(update))
            if self.temptrack.get():
                self.temp.set(round_sig_figs(grab_temp(self.dev_temp.get(), self.chan_temp.get(), num=250), sig_fig=4))
            self.w.after(600 - int((time.time() - self.time_start) % 600), self.update)  # update on a rate of 100/min

    def run(self):
        # hand off to functions that get inputs and run measurements
        if self.run_type.get() == "autotemp":
            self.run_autotemp()
        elif self.run_type.get() == "single":
            self.run_single()

    def run_autotemp(self):
        self.update(stop=True)
        at = AutoTemp(save_folder_path=self.save_path.get() + "/AutoTemp",
                      dev_signal=self.dev_signal.get() + '/' + self.chan_signal.get(), vpp=self.vpp.get(),
                      dev_temp=self.dev_temp.get() + '/' + self.chan_temp.get(),
                      sample_name=self.sample_name.get(), bounds=[self.boundL.get(), self.boundU.get()],
                      t=self.t.get())
        match self.method.get():
            case 'P':
                # at.auto_temp_pulse(delay=10, temp_step=self.tempstep.get(),
                #                    temp_repeats=self.repeats.get(), runs=self.runs.get())
                at.auto_pulse(time_between=1, repeats=self.repeats.get(), runs=self.runs.get())
            case 'A':
                at.auto_temp_adaptive(tolerance=5, start_guess=5e2, start_delta=1e2, temp_step=self.tempstep.get(),
                                      temp_repeats=self.repeats.get())
            case 'S':
                # at.auto_temp_sweep(freqstep=self.freqstep.get(), temp_step=self.tempstep.get(),
                #                    temp_repeats=self.repeats.get(), temp_start=self.tempL.get(),
                #                    temp_stop=self.tempU.get(), GUI=self)
                at.auto_sweep(freqstep=self.freqstep.get(), repeats=self.repeats.get())
        self.w.after(600, notify_finish)  # extra notify for autotemp as I may b distracted!
        at.close()

        notify_finish()
        self.w.after(1, self.update, False)

    def run_single(self):
        self.update(stop=True)
        match self.method.get():
            case 'P':
                measure_pulse_decay(self.dev_signal.get() + '/' + self.chan_signal.get(), runs=self.runs.get(),
                                    delay=10, t=self.t.get(), GUI=self)
            case 'A':
                measure_adaptive(self.dev_signal.get() + '/' + self.chan_signal.get(), vpp=self.vpp.get(),
                                 tolerance=self.freqstep.get(), start_guess=600, deltainit=1e2,
                                 bounds=[self.boundL.get(), self.boundU.get()])
            case 'S':
                measure_sweep(freq=[self.boundL.get(), self.boundU.get()], freqstep=self.freqstep.get(),
                              t=self.t.get(), vpp=self.vpp.get(),
                              devchan=self.dev_signal.get() + '/' + self.chan_signal.get(), GUI=self)
        # save output and fit after finishing single run
        if self.temptrack.get():
            self.temp.set(
                round_sig_figs(grab_temp(self.dev_temp.get(), self.chan_temp.get(), num=10000), sig_fig=4))
        self.resave_output()
        if self.fit.get():
            self.w.after(1, fit, "outputs/output.txt", True)  # todo cutoff

        notify_finish()
        self.w.after(1, self.update, False)

    def calibrate(self):
        self.update(stop=True)
        calibrate(mode='dual', c1=self.dev_signal.get() + '/' + self.chan_signal.get(),
                  c2=self.dev_temp.get() + '/' + self.chan_temp.get())
        self.w.after(1, self.update, False)

    def resave_output(self):
        method = self.method.get()
        match method:
            case 'S':
                method = method + f'{self.vpp.get():.2g}' + 'V'
            case 'P':
                method = method + str(int(self.runs.get()))
        resave_output(method=method, save_path=self.save_path.get(), temperature=self.temp.get(),
                      sample=self.sample_name.get(), copy=False)

    def manual_peak(self):
        manual_peak_auto(save_path=self.save_path.get(), cutoff=[0, 1], sample=self.sample_name.get(),
                         printer=self.Writer)
        # # manual_peak(save_path=save_path + r"\AutoTemp", cutoff=[0.05, 0.6])
        # resave(save_path + r"\AutoTemp")

    def wire(self):
        wireplot_manager(self.save_path.get(), mode='temp', printer=self.Writer)

    def select_path(self):
        self.save_path.set(filedialog.askdirectory())
        self.entry_path.xview_moveto(1)

    def close(self):
        self.Writer.close()
        matplotlibclose("all")
        self.w.destroy()
        # print("Closed")
        # quit()


class Writer:
    def __init__(self, text_holder):
        self.text_holder = text_holder
        self.writestore = sys.stdout.write  # store it for use later when closing
        sys.stdout.write = self.write  # redirect "print" to the GUI

    def write(self, s):
        # s.split("\n")
        #  for loop over all those splits to make sure it is handled correctly?
        if s == '\n' or s == '':  # todo empty string ignore?
            return
        if s[0:1] == '\r':  # todo this currently deletes the last tqdm line if the next line begins with \r
            s = s[1:]
            last_line = self.text_holder.get("end-2c linestart", "end-2c lineend")
            if len(last_line) >= 7:
                if not (last_line[0] == "[" and last_line[3] == ":" and last_line[6] == "]"):
                    self.text_holder.delete('end-2l', 'end-1l')  # replace previous line if it was tqdm
            self.text_holder.insert(tk.END, s + '\n')
        else:
            if s[0:1] == '\n':
                s = s[1:]
            self.text_holder.insert(tk.END,
                                    '[' + ':'.join([str(e).zfill(2) for e in time.localtime()[3:5]]) + ']' + s + '\n')
        self.text_holder.see(tk.END)

    def close(self):
        sys.stdout.write = self.writestore


def buttonfunc():
    print("button")
    scatter3d()
    # fft_magnitude_and_phase()
    # # a = input("done?")
    # time_frequency_spectrum2electricboogaloo(sigma=1e2)
    # time_frequency_spectrum2electricboogaloo(sigma=1e3)


def audio_raw():
    play_mic_signal()


def audio_spectra():
    play_spectra(filedialog.askopenfilename())


def audio_stop():
    play_stop()
