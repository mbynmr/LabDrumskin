from measurement import measure, measure_pulse
from fitting import fit


# in anaconda prompt (pinned)
# >conda activate LabDrumskin
# if you want to go to the folder then
# >cd PycharmProjects
# >cd LabDrumskin


def main():
    # measure_pulse()  # make this work!
    measure(freq=[1800, 3600], freqstep=5, t=1)
    # fit("outputs/2022_11_7_S_C5_25.5_00.txt", [0.0, 1])
    path = "C:/Users/mbynmr/OneDrive - The University of Nottingham/Documents" \
           "/Shared - Mechanical Vibrations of Ultrathin Films/Lab/data/may all have wax on them" \
           "/temps measured on C5/2022_11_7_S_C5_25_00.txt"
    # path = "outputs/2022___.txt"
    # fit(path, [0.0, 1])


if __name__ == '__main__':
    main()
