from measurement import measure, measure_pulse_decay
from fitting import fit
from my_tools import resave_output


# in anaconda prompt (pinned)
# >conda activate LabDrumskin
# if you want to go to the folder then
# >cd PycharmProjects
# >cd LabDrumskin


def main():
    save_path = "C:/Users/mbynmr/OneDrive - The University of Nottingham/Documents" \
           "/Shared - Mechanical Vibrations of Ultrathin Films/Lab/data/Temperature Sweeps"
    method = "S"
    method = "P"
    match method:
        case "S":
            measure(freq=[2600, 1950], freqstep=5, t=1)
            output_file = resave_output(method="S", save_path=save_path)
            fit("outputs/output.txt", [0, 1])
        case "P":
            measure_pulse_decay("Dev2/ai0")
            output_file = resave_output(method="P", save_path=save_path)
            fit("outputs/output.txt", [0.25, 0.75])
    # fit(f"{save_path}/2022_11_18_14_59_S_C11_28.5.txt", [0, 1])
    # fit(output_file, [0.0, 1])


if __name__ == '__main__':
    main()
