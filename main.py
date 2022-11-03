from measurement import measure, measure_pulse
from fitting import fit


# in anaconda prompt (pinned)
# >conda activate LabDrumskin
# if you want to go to the folder then
# >cd PycharmProjects
# >cd LabDrumskin


def main():
    # measure_pulse()  # make this work!
    measure(freq=[2400, 3200], freqstep=5, t=1)
    # fit("outputs/S-52.5-C4_00.txt", [0.0, 1])


if __name__ == '__main__':
    main()
