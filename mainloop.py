import time
import tkinter as tk
from tkinter import filedialog, scrolledtext
import sys


from measurement import measure_sweep, measure_pulse_decay, measure_adaptive
from automation import AutoTemp
from my_tools import resave_output, resave_auto
from aggregation import aggregate, colourplot, manual_peak, manual_peak_auto
from aggregatestuff import resave
from fitting import fit, find_peaks


class Main:
    """
    big main class for GUI (which deals with everything else)
    """

    def __init__(self):
        # mainwindowobject is self.w
        self.w = tk.Tk(className='Main')
        self.w.geometry("600x400")  # this is the worst. i hate it.

        # set up variables that can be used in buttons
        self.method = tk.StringVar(self.w, value='P')
        self.run_type = tk.StringVar(self.w, value='single')
        self.fit = tk.BooleanVar(self.w, value=True)
        self.sample_name = tk.StringVar(self.w, value='SAMPLENAME')
        self.save_path = tk.StringVar(self.w, value='outputs')
        self.t = tk.DoubleVar(self.w, value=0.2)
        self.runs = tk.DoubleVar(self.w, value=100)
        self.freqstep = tk.DoubleVar(self.w, value=5)
        self.boundU = tk.DoubleVar(self.w, value=5000)
        self.boundL = tk.DoubleVar(self.w, value=50)
        self.vpp = tk.DoubleVar(self.w, value=10)
        self.dev_signal = tk.StringVar(self.w, value='Dev2/ai0')  # todo dropdown of devices? xD
        self.dev_temp = tk.StringVar(self.w, value='Dev2/ai1')

        # widgets before mainloop
        self.widgets()

        # print command redirect
        self.print_box = scrolledtext.ScrolledText(self.w, width=42, height=13)
        self.print_box.place(relx=0.01, rely=0.4)
        self.writestore = sys.stdout.write  # store it for use later when closing
        sys.stdout.write = self.write  # redirect "print" to the GUI

        self.w.mainloop()

    def widgets(self):
        # some buttons
        tk.Button(self.w, text='Measure', command=self.run).place(relx=0.05, rely=0.125)
        tk.Button(self.w, text='Pause', command=self.pause).place(relx=0.2, rely=0.075)
        tk.Button(self.w, text='Stop', command=self.stop).place(relx=0.2, rely=0.175)
        tk.Button(self.w, text='Recover (save) last measurement', command=self.resave_output).place(relx=0.1, rely=0.3)
        tk.Button(self.w, text='Force save auto', command=self.resave_auto).place(relx=0.5, rely=0.3)
        tk.Button(self.w, text='Manual peaks', command=buttonfunc).place(relx=0.7, rely=0.3)
        tk.Button(self.w, text='Close', command=self.close).place(relx=0.1, rely=0.925)

        # method options
        tk.Radiobutton(self.w, text='Sweep', variable=self.method, value='S').place(relx=0.3, rely=0.05)
        tk.Radiobutton(self.w, text='Adapt', variable=self.method, value='A').place(relx=0.3, rely=0.125)
        tk.Radiobutton(self.w, text='Pulse', variable=self.method, value='P').place(relx=0.3, rely=0.2)

        # method options for multiple/single
        tk.Radiobutton(self.w, text='Single run', variable=self.run_type, value='single').place(relx=0.4, rely=0.075)
        tk.Radiobutton(self.w, text='Autotemp', variable=self.run_type, value='autotemp').place(relx=0.4, rely=0.175)

        # toggle fit view after
        tk.Checkbutton(self.w, text="Fit after?", variable=self.fit).place(relx=0.55, rely=0.075)

        x = 0.05  # x shift for lots of things at once
        # text entries with labels
        tk.Entry(self.w, textvariable=self.sample_name).place(relx=0.7 + x, rely=0.1)
        tk.Label(self.w, text="Sample name").place(relx=0.7 + x, rely=0.05)
        tk.Entry(self.w, textvariable=self.save_path).place(relx=0.7 + x, rely=0.2)
        tk.Label(self.w, text="Save path").place(relx=0.7 + x, rely=0.15)
        tk.Button(self.w, text='Path', command=self.select_path).place(relx=0.88 + x, rely=0.19)
        tk.Entry(self.w, textvariable=self.t).place(relx=0.7 + x, rely=0.45)
        tk.Label(self.w, text="Measure time").place(relx=0.56 + x, rely=0.45)
        tk.Entry(self.w, textvariable=self.boundL).place(relx=0.6 + x, rely=0.5)
        tk.Entry(self.w, textvariable=self.boundU).place(relx=0.7 + x, rely=0.5)
        tk.Label(self.w, text="< f <").place(relx=0.645 + x, rely=0.5)
        tk.Entry(self.w, textvariable=self.runs).place(relx=0.7 + x, rely=0.6)
        tk.Label(self.w, text="Repeats (pulse)").place(relx=0.555 + x, rely=0.6)
        tk.Entry(self.w, textvariable=self.freqstep).place(relx=0.7 + x, rely=0.7)
        tk.Label(self.w, text="f step (sweep)").place(relx=0.55 + x, rely=0.7)
        tk.Entry(self.w, textvariable=self.vpp).place(relx=0.7 + x, rely=0.75)
        tk.Label(self.w, text="Vpp (sweep)").place(relx=0.58 + x, rely=0.75)
        tk.Entry(self.w, textvariable=self.dev_signal).place(relx=0.7 + x, rely=0.85)
        tk.Label(self.w, text="Signal device").place(relx=0.57 + x, rely=0.85)
        tk.Entry(self.w, textvariable=self.dev_temp).place(relx=0.7 + x, rely=0.9)
        tk.Label(self.w, text="Temp device").place(relx=0.5725 + x, rely=0.9)

    def write(self, s):
        # todo s.split("\n")
        #  for loop over all those splits to make sure it is handled correctly
        if s != '\n':
            if s[0:1] == '\r':
                s = s[1:]
                self.print_box.delete('end-2l', 'end-1l')  # todo this currently removes ANY line before, even if i=0
                self.print_box.insert(tk.END, s + '\n')
            else:
                self.print_box.insert(tk.END,
                                      '[' + ':'.join([str(e).zfill(2) for e in time.localtime()[3:5]]) + ']' + s + '\n')
            self.print_box.see("end")

    def select_path(self):
        root = tk.Tk(className='Path Selection')
        root.withdraw()
        self.save_path.set(filedialog.askdirectory())

    def stop(self):
        print("stop")

    def pause(self):
        print("pause")

    def run(self):
        # get inputs as they are needed and run measurements

        if self.run_type.get() == "autotemp":
            at = AutoTemp(save_folder_path=self.save_path.get() + r"\AutoTemp",
                          dev_signal=self.dev_signal.get(), vpp=self.vpp.get(), dev_temp=self.dev_temp.get(),
                          sample_name=self.sample_name.get(), bounds=[self.boundL.get(), self.boundU.get()],
                          t=self.t.get())
            match self.run_type.get():
                case 'P':
                    at.auto_temp_pulse(delay=10, temp_step=5, temp_repeats=10, runs=self.runs.get())
                    at.auto_pulse(delay=10, time_between=30, repeats=1, runs=self.runs.get(), temp='Y')  # todo temp
                case 'A':
                    at.auto_temp_adaptive(tolerance=5, start_guess=5e2, start_delta=1e2, temp_step=2.5, temp_repeats=3)
                case 'S':
                    at.auto_temp_sweep(freqstep=50, temp_step=2.5, temp_repeats=2)
            at.close()
        elif self.run_type.get() == "single":
            match self.method.get():
                case 'P':
                    measure_pulse_decay(self.dev_signal.get(), runs=self.runs.get(), delay=10, t=self.t.get(),
                                        printer=self)
                case 'A':
                    measure_adaptive(self.dev_signal.get(), vpp=self.vpp.get(), tolerance=self.freqstep.get(),
                                     start_guess=600, deltainit=1e2, bounds=[self.boundL.get(), self.boundU.get()])
                case 'S':
                    measure_sweep(freq=[self.boundL.get(), self.boundU.get()], freqstep=self.freqstep.get(),
                                  t=self.t.get(), vpp=self.vpp.get(), devchan=self.dev_signal.get(), printer=self)

        # save output and fit after finishing run
        self.resave_output()
        if self.fit.get():
            fit(file_name_and_path="outputs/output.txt", copy=True)  # todo cutoff

    def resave_output(self):
        resave_output(save_path=self.save_path.get(), sample_name=self.sample_name.get(), method=self.method.get(),
                      temperature=20, copy=True)

    def resave_auto(self):
        resave_auto(save_path=self.save_path.get() + r"\AutoTemp", sample_name=self.sample_name.get(),
                    method=self.method.get(), manual=False)

        # # manual_peak(save_path=save_path + r"\AutoTemp", cutoff=[0.05, 0.6])
        # manual_peak_auto(save_path=save_path + r"\AutoTemp", cutoff=[0.05, 0.25])
        # resave(save_path + r"\AutoTemp")

    def close(self):
        sys.stdout.write = self.writestore
        self.w.destroy()


def buttonfunc():
    print("button")


def oldmain():
    save_path = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" + \
                r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\PDMS\new piezos"
    # save_path = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" + \
    #             r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\PSY/1.5 percent"
    # dev2 is blue and is the small blue (1 layer breadboard) setup
    dev_sign = "Dev2/ai0"
    dev_temp = "Dev2/ai1"
    # dev1 is cream and is the large cream (2 layer breadboard) setup
    # dev_sign = "Dev1/ai0"
    # dev_temp = "Dev1/ai1"
    # method = "agg"
    method = "chooser"
    # method = "A"
    #method = "S"  # sweep
    # shortcut to run is F5 or press the green button
    method = "P"  # pulse
    # method = "AutoTemp_" + method
    # method = "other_" + method

    # find_peaks()
    match method.split("_")[0]:
        case "S":
            measure_sweep(freq=[50, 3600], freqstep=5, t=0.2, vpp=10, devchan=dev_sign)
            resave_output(method="S", save_path=save_path, copy=True)
            fit("outputs/output.txt", [0.05, 0.5])
        case "A":
            measure_adaptive(dev_sign, vpp=5, tolerance=5, start_guess=600, deltainit=1e2, bounds=[100, 900])
            resave_output(method="A", save_path=save_path, copy=True)
            fit("outputs/output.txt", [0.05, 0.5])
        case "P":
            measure_pulse_decay(dev_sign, runs=100, delay=10)  # runs is how many pulses to average over
            resave_output(method="P", save_path=save_path, copy=True)
            fit("outputs/output.txt", [0.05, 0.5])
        case "AutoTemp":
            at = AutoTemp(save_folder_path=save_path + r"\AutoTemp", dev_signal=dev_sign, vpp=10, dev_temp=dev_temp)
            match method.split("_")[-1]:
                case "S":
                    at.auto_temp_sweep(freqstep=50, temp_step=2.5, temp_repeats=2)
                case "A":
                    at.auto_temp_adaptive(tolerance=5, start_guess=5e2, start_delta=1e2, temp_step=2.5, temp_repeats=3)
                case "P":
                    # at.auto_temp_pulse(delay=10, temp_step=5, temp_repeats=10)
                    at.auto_pulse(delay=10, time_between=30, repeats=100, runs=66)
                case _:
                    at.sample_name = "_"
                    at.calibrate()
            at.close()
        case "agg":
            aggregate()
        case "chooser":
            # manual_peak(save_path=save_path + r"\AutoTemp", cutoff=[0.05, 0.6])
            manual_peak_auto(save_path=save_path + r"\AutoTemp", cutoff=[0.05, 0.25])
        case _:
            resave(save_path + r"\AutoTemp")
            # resave_auto(save_path=save_path + r"\AutoTemp")
            # colourplot()
