from testIO import test
from measurement import measure

# in anaconda prompt (pinned)
# conda activate LabDrumskin
# pip freeze


def main():
    # test()
    measure(freq=[900, 920], freqstep=5, t=2)
    # print("work")


if __name__ == '__main__':
    main()
