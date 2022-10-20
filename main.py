# from testIO import waveform_generators, picoscope, picoscope_check
# from measurement import measure, measure_pulse
from analysis import peak_finder_averages, peak_finder_with_derivatives

# in anaconda prompt (pinned)
# >conda activate LabDrumskin
# if you want to go to the folder then
# >cd PycharmProjects
# >cd LabDrumskin

# todo project for Python MPAG:
#  Write a function that controls what frequency we are looking at for fastest possible acquisition.
#  Feed it the freqs bounds and it makes guesses, changing the measurements adaptively to measure for less time overall.
#  While this may not be the final method we will use to find the resonant frequency, it is the method we are using for
#  now. Having 5Hz resolution and sweeping from 50 to 2500 with 2s time period takes over 15 minutes. Given some prior
#  knowledge of where the peak will likely be, this can be reduced to a few minutes. Lowering the measurement time
#  period (and so signal to noise) is currently an option but the films will get thinner and less loud (literally) in
#  response to driving, meaning a lower signal to noise ratio, so tests will take longer in the future.
#  We will need to trust 100% that it hasn't picked up on a peak that isn't the resonant frequency of the film, as
#  visually looking at the graphs each time is a waste of time.
#  This isn't the bottleneck of the whole process but it would be nice to speed up still.


def main():
    # peak_finder_averages()
    peak_finder_with_derivatives()
    # picoscope_check()
    # picoscope()
    # measure(freq=[900, 1500], freqstep=5, t=2)
    # measure_pulse()  # this no works :c


if __name__ == '__main__':
    main()
