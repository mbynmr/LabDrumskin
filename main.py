from measurement import measure, measure_pulse_decay, measure_adaptive
from automation import AutoMeasure
from fitting import fit
from my_tools import resave_output


def main():
    save_path = r"C:\Users\mbynmr\OneDrive - The University of Nottingham\Documents" +\
                r"\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\Temperature Sweeps"
    # method = "S"
    method = "A"
    # method = "P"
    # method = "Auto"
    match method:
        case "S":
            measure(freq=[3200, 800], freqstep=5, t=1, devchan="Dev2/ai0")
            output_file = resave_output(method="S", save_path=save_path)
            fit("outputs/output.txt", [0, 1])
        case "A":
            measure_adaptive("Dev2/ai0", tolerance=5)
            output_file = resave_output(method="A", save_path=save_path)
            fit("outputs/output.txt", [0, 1])
        case "P":
            measure_pulse_decay("Dev2/ai0")
            output_file = resave_output(method="P", save_path=save_path)
            fit("outputs/output.txt", [0.25, 0.75])
        case "Auto":
            am = AutoMeasure(save_folder_path=save_path, sample_name="C15", dev_signal="Dev2/ai0", dev_temp="Dev2/ai1")
            am.auto_temp_pulse(time_between_measurement_starts=60, total_measurements=60)
            am.close()

    # fit(output_file, [0.0, 1])


if __name__ == '__main__':
    main()
