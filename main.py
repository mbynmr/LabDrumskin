from measurement import measure_sweep, measure_pulse_decay, measure_adaptive
from automation import AutoTemp
from fitting import fit
from my_tools import resave_output, resave_auto


def main():
    method = "none"
    save_path = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" +\
                r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\Temperature Sweeps"
    dev_signal = "Dev2/ai0"
    dev_temp = "Dev2/ai1"
    method = "S"
    # method = "A"
    method = "P"
    method = "AutoTemp_" + method
    match method.split("_")[0]:
        case "S":
            measure_sweep(freq=[50, 2000], freqstep=25, t=0.5, vpp=5, devchan=dev_signal)
            output_file = resave_output(method="S", save_path=save_path)
            fit("outputs/output.txt", [0, 1])
        case "A":
            measure_adaptive(dev_signal, vpp=5, tolerance=5, start_guess=800, deltainit=1e2, bounds=[100, 3.5e3])
            output_file = resave_output(method="A", save_path=save_path)
            fit("outputs/output.txt", [0, 1], copy=False)
        case "P":
            measure_pulse_decay(dev_signal, runs=33, delay=30)
            output_file = resave_output(method="P", save_path=save_path)
            fit("outputs/output.txt", [0.1, 0.7])
        case "AutoTemp":
            at = AutoTemp(save_folder_path=save_path, dev_signal=dev_signal, vpp=10, dev_temp=dev_temp)
            match method.split("_")[-1]:
                case "S":
                    at.auto_temp_sweep(temp_step=2.5, temp_repeats=2, freq=[2000, 2600], freqstep=5)
                case "A":
                    at.auto_temp_adaptive(tolerance=5, start_guess=700, start_delta=1e2, bounds=[1e2, 1e3],
                                          temp_step=2.5, temp_repeats=3)
                case "P":
                    at.auto_temp_pulse(temp_step=2.5, temp_repeats=3, cutoff=[0.1, 0.7], delay=30)
                case _:
                    at.sample_name = "_"
                    at.calibrate()
            at.close()
        case _:
            resave_auto(save_path=save_path)


if __name__ == '__main__':
    main()
