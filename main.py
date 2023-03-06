from measurement import measure_sweep, measure_pulse_decay, measure_adaptive
from automation import AutoTemp
from fitting import fit
from my_tools import resave_output, resave_auto
from aggregation import aggregate, colourplot, manual_peak


def main():
    method = "none"
    save_path = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" + \
                r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\PIM"
    dev_signal = "Dev2/ai0"
    dev_temp = "Dev2/ai1"
    method = "agg"
    # method = "chooser"
    method = "S"
    # method = "A"
    # method = "P"
    # method = "AutoTemp_" + method
    # method = "other_" + method'

    match method.split("_")[0]:
        case "S":
            measure_sweep(freq=[50, 20000], freqstep=50, t=0.2, vpp=10, devchan=dev_signal)
            output_file = resave_output(method="S", save_path=save_path)
            fit("outputs/output.txt", [0, 1])
        case "A":
            measure_adaptive(dev_signal, vpp=5, tolerance=5, start_guess=600, deltainit=1e2, bounds=[100, 900])
            output_file = resave_output(method="A", save_path=save_path)
            fit("outputs/output.txt", [0, 1], copy=False)
        case "P":
            measure_pulse_decay(dev_signal, runs=33, delay=10)
            output_file = resave_output(method="P", save_path=save_path)
            fit("outputs/output.txt", [0.05, 1])
        case "AutoTemp":
            at = AutoTemp(save_folder_path=save_path + r"\AutoTemp", dev_signal=dev_signal, vpp=10, dev_temp=dev_temp)
            bounds = [float(input("lower freq:")), float(input("upper freq:"))]
            match method.split("_")[-1]:
                case "S":
                    at.auto_temp_sweep(bounds=bounds, freqstep=50, temp_step=2.5, temp_repeats=2)
                case "A":
                    at.auto_temp_adaptive(tolerance=5, start_guess=5e2, start_delta=1e2, bounds=bounds,
                                          temp_step=2.5, temp_repeats=3)
                case "P":
                    at.auto_temp_pulse(bounds=bounds, delay=10, temp_step=2.5, temp_repeats=3)
                case _:
                    at.sample_name = "_"
                    at.calibrate()
            at.close()
        case "agg":
            aggregate()
        case "chooser":
            manual_peak(save_path=save_path + r"\AutoTemp", cutoff=[0.45, 0.6])
        case _:
            resave_auto(save_path=save_path + r"\AutoTemp")
            # colourplot()


if __name__ == '__main__':
    main()
