# from measurement import measure, measure_pulse, measure_adaptive
from fitting import fit


# in anaconda prompt (pinned)
# >conda activate LabDrumskin
# if you want to go to the folder then
# >cd PycharmProjects
# >cd LabDrumskin


def main():
    # mpag()
    # peak_finder_averages()
    # peak_finder_with_derivatives()
    # picoscope_check()
    # picoscope()
    measure(freq=[1500, 2500], freqstep=5, t=0.5)
    # measure_adaptive(freq=[900, 1500], t=2)
    # measure_pulse()  # make this work!
    # fit("outputs/2022 10 27 first good data/S-C2_01.txt", [0.0, 1])


if __name__ == '__main__':
    main()
