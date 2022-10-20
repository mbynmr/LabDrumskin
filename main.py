from testIO import test
from measurement import measure, measure_pulse

# in anaconda prompt (pinned)
# conda activate LabDrumskin
# pip freeze


def main():
    # test()
    measure(freq=[900, 1500], freqstep=5, t=2)
    # measure_pulse()  # todo check if this works!
    # print("work")


if __name__ == '__main__':
    main()
