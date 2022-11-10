from measurement import measure
from fitting import fit


# in anaconda prompt (pinned)
# >conda activate LabDrumskin
# if you want to go to the folder then
# >cd PycharmProjects
# >cd LabDrumskin


def main():
    # measure_pulse()  # make this work!
    save_path = "C:/Users/mbynmr/OneDrive - The University of Nottingham/Documents" \
           "/Shared - Mechanical Vibrations of Ultrathin Films/Lab/data/"
    output_file = measure(freq=[1800, 3800], freqstep=5, t=1, save_path=save_path)
    fit("outputs/output.txt", [0.0, 1])
    # fit(output_file, [0.0, 1])
    # fit("C:/Users/mbynmr/OneDrive - The University of Nottingham/Documents/"
    #     "Shared - Mechanical Vibrations of Ultrathin Films/Lab/data/2022_11_09_16_17_S_C8_28.5.txt", [0.0, 1])


if __name__ == '__main__':
    main()
