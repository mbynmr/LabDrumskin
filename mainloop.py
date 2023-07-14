import tkinter as tk


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
        # mainwindowobject is w
        self.w = tk.Tk(className='Main')
        self.w.geometry("600x400")  # this is the worst. i hate it.

        self.method = tk.StringVar(self.w, value='P')
        self.run_type = tk.StringVar(self.w, value='single')
        self.save_path = tk.StringVar(self.w, value='outputs')
        self.dev_signal = tk.StringVar(self.w, value='Dev2/ai0')  # todo dropdown of devices? xD
        self.dev_temp = tk.StringVar(self.w, value='Dev2/ai1')
        self.vpp = tk.DoubleVar(self.w, value=10)
        self.sample_name = tk.StringVar(self.w, value='SAMPLENAME')
        self.boundU = tk.DoubleVar(self.w, value=5000)
        self.boundL = tk.DoubleVar(self.w, value=50)

        # widgets before mainloop
        self.widgets()
        self.mainloop()

    def widgets(self):
        # some buttons
        tk.Button(self.w, text='Run', command=self.run).place(relx=0.1, rely=0.1)
        tk.Button(self.w, text='resave output', command=buttonfunc).place(relx=0.1, rely=0.4)
        tk.Button(self.w, text='resave auto', command=buttonfunc).place(relx=0.1, rely=0.5)
        tk.Button(self.w, text='Manual peaks', command=buttonfunc).place(relx=0.1, rely=0.6)
        tk.Button(self.w, text='Close', command=self.close).place(relx=0.1, rely=0.9)

        # method options
        tk.Radiobutton(self.w, text='Sweep', variable=self.method, value='S').place(relx=0.3, rely=0.1)
        tk.Radiobutton(self.w, text='Adapt', variable=self.method, value='A').place(relx=0.3, rely=0.175)
        tk.Radiobutton(self.w, text='Pulse', variable=self.method, value='P').place(relx=0.3, rely=0.25)

        # method options for multiple
        tk.Radiobutton(self.w, text='single run', variable=self.run_type, value='single').place(relx=0.4, rely=0.15)
        tk.Radiobutton(self.w, text='autotemp', variable=self.run_type, value='autotemp').place(relx=0.4, rely=0.2)

        # text entries
        tk.Entry(self.w, textvariable=self.save_path).place(relx=0.5, rely=0.175)
        tk.Entry(self.w, textvariable=self.dev_signal).place(relx=0.7, rely=0.7)
        tk.Entry(self.w, textvariable=self.dev_temp).place(relx=0.7, rely=0.75)

        tk.Entry(self.w, textvariable=self.vpp).place(relx=0.7, rely=0.75)
        tk.Entry(self.w, textvariable=self.sample_name).place(relx=0.7, rely=0.75)
        tk.Entry(self.w, textvariable=self.boundU).place(relx=0.7, rely=0.75)
        tk.Entry(self.w, textvariable=self.boundL).place(relx=0.7, rely=0.75)

    def mainloop(self):
        # go!
        self.w.mainloop()

    def run(self):
        # get inputs as they are needed

        if self.run_type.get() == "autotemp":
            at = AutoTemp(save_folder_path=self.save_path.get() + r"\AutoTemp",
                          dev_signal=self.dev_signal.get(), vpp=self.vpp.get(), dev_temp=self.dev_temp.get(),
                          sample_name=self.sample_name.get(), bounds=[self.boundL.get(), self.boundU.get()])

            # measure here
            at.close()
        elif self.run_type.get() == "single":
            match self.method.get():
                case 'P':
                    print("pulsing")
                case 'A':
                    print("adaptive")
                case 'S':
                    print("sweeping")

    def close(self):
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
