import numpy as np
import matplotlib.pyplot as plt
import time


# todo methods for finding the peak:
#  simply taking the maximum is not sufficient as noise may be larger, but there could still be a peak trend
#  find local slope? Do 2nd derivative?
#  match to a known shape? This will need a secondary check to ensure
#  check if local mean is somewhat larger than global mean?
#  check if local volume is somewhat larger than average expected volume? Is this the same as the means method?


def peak_finder_averages():
    """
    Prints the frequency where the largest amplitude
    """

    data = np.loadtxt("outputs/python data/J_002.txt")
    # data.shape = (101, 2) as an example
    # data[:, 0] = frequencies  (x axis)
    # data[:, 1] = amplitudes   (y axis)
    a = np.array(data[:, 1])  # make a copy of the amplitudes

    i = 0  # todo find a way to not have this!
    found = False
    while not found:
        if np.all(a == np.nan):  # if all points are searched
            break  # this will be useful for the version of the code that will measure then check for a peak repeatedly
        i = np.argmax(a)
        print(f"Searching f = {data[i, 0]} Hz")

        if np.nanmean(a[max(0, i - 10):min(len(a) + 1, i + 10)]) > np.nanmean(a[:]):
            # when the local amplitudes around index i are larger than the global average
            found = True
        else:
            a[i] = np.nan  # make sure this frequency doesn't come up again and doesn't contribute to any averages

    if found:
        print(f"Peak is f = {data[i, 0]} Hz")
        plt.plot(data[:, 0], data[:, 1])
        plt.plot([data[i, 0], data[i, 0]], [min(data[:, 1]), max(data[:, 1])])
        plt.show()
    else:
        print("can't find a peak!")  # shouldn't need to be used in this version!


def peak_finder_with_derivatives():
    data = np.loadtxt("outputs/python data/J_002.txt")
    a = np.array(data[:, 1])  # make a copy of the amplitudes

    found = False
    while not found:
        if np.all(a == np.nan):  # if all points are searched
            break  # this will be useful for the version of the code that will measure then check for a peak repeatedly

        # d1 = gradient of a  # todo assuming equal spacing of x axis!
        d1 = np.gradient(a)
        # d2 = gradient of d1
        d2 = np.gradient(d1)
        # find every set of two points between which d1 changes sign
        change_indexes = np.where(np.sign(d1[:-1]) == np.sign(d1[1:]), 0, 1)
        # this is the same length as d2 so [1, 1, -1, -1, 1] is [0, 0, 1, 0, 1]. Removing the first element centres it
        change_indexes = np.nonzero(change_indexes + np.roll(change_indexes, -1))
        # now [1, 1, -1, -1, 1] is [0, 1, 1, 1, 1], so every index next to a change is included.

        i = np.argmax(a[change_indexes])  # todo carry on from here tomorrow!

        # todo interpolate for frequency where d2 = 0 exactly?

        if not found:  # todo replace with some condition that means it has to be a peak and not just noise!
            found = True
            print(f"Peak is f = {data[i, 0]} Hz")
        else:
            a[i] = np.nan  # make sure this frequency doesn't come up again

    if found:
        plt.plot(data[:, 0], data[:, 1])
        plt.plot([data[i, 0], data[i, 0]], [min(data[:, 1]), max(data[:, 1])])
        plt.show()
    else:
        print("can't find a peak!")  # shouldn't need to be used in this version!
