from measurement import measure_sweep, measure_pulse_decay, measure_adaptive
from automation import AutoTemp
from my_tools import resave_output, resave_auto
from aggregation import aggregate, colourplot, manual_peak, manual_peak_auto
from aggregatestuff import resave
from fitting import fit, find_peaks


def main():
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


if __name__ == '__main__':
    main()
