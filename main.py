from measurement import measure, measure_pulse_decay, measure_adaptive
from automation import AutoTemp
from fitting import fit
from my_tools import resave_output  # , resave_auto


def main():
    save_path = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" +\
                r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\Temperature Sweeps"
    dev_signal = "Dev2/ai0"
    method = "S"
    method = "A"
    method = "P"
    method = "AutoTemp"
    # method = "dummy"
    match method:
        case "S":
            measure(freq=[2100, 2300], freqstep=5, t=1, devchan=dev_signal)
            output_file = resave_output(method="S", save_path=save_path)
            fit("outputs/output.txt", [0, 1])
        case "A":
            measure_adaptive(dev_signal, tolerance=5, start_guess=1500)
            output_file = resave_output(method="A", save_path=save_path)
            fit("outputs/output.txt", [0, 1], copy=False)
        case "P":
            measure_pulse_decay(dev_signal, runs=100)
            output_file = resave_output(method="P", save_path=save_path)
            fit("outputs/output.txt", [0.25, 0.6])
        case "AutoTemp":
            at = AutoTemp(save_folder_path=save_path, dev_signal=dev_signal, dev_temp="Dev2/ai1", sample_name="EA0")
            at.auto_temp_pulse(temp_start=45, temp_stop=35, temp_step=5, temp_repeats=2, cutoff=[0.25, 0.6])
            # resave_auto(save_path=save_path, sample_name="EA0", method="P")
            at.close()
        # case "dummy":
        #     at = AutoTemp(save_folder_path=save_path, dev_signal=dev_signal, dev_temp="Dev2/ai1", sample_name="none")
        #     at.calibrate()


if __name__ == '__main__':
    main()
