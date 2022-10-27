from measurement import measure, measure_pulse, measure_adaptive
# from analysis import peak_finder_averages, peak_finder_with_derivatives
# from mpag import mpag


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
    measure(freq=[900, 3500], freqstep=5, t=2)
    # measure_adaptive(freq=[900, 1500], t=2)
    # measure_pulse()  # make this work!


if __name__ == '__main__':
    main()
