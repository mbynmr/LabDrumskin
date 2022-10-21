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
        i = np.argwhere(a == np.nanmax(a)).flatten()  # todo flatten isn't elegant.
        print(f"Searching f = {data[i, 0]} Hz")

        if np.nanmean(a[max(0, i - 10):min(len(a) + 1, i + 10)]) > np.nanmean(a[:]):
            # when the local amplitudes around index i are larger than the global average
            found = True
        else:
            a[i] = np.nan  # make sure this frequency is ignored and doesn't contribute to any averages

    if found:
        print(f"Peak is f = {data[i, 0]} Hz")
        plt.plot(data[:, 0], data[:, 1])
        plt.plot([data[i, 0], data[i, 0]], [min(data[:, 1]), max(data[:, 1])])
        plt.show()
    else:
        print("can't find a peak!")  # shouldn't need to be used in this version!


def peak_finder_with_derivatives():
    data = np.loadtxt("outputs/python data/J_000.txt")
    a = np.array(data[:, 1])  # make a copy of the amplitudes

    found = False
    likely_correct = False
    while not found:
        # todo assuming equal spacing of x axis!
        d1 = np.gradient(a)
        d1_indexes = [0, *np.where(np.sign(d1[:-1]) == np.sign(d1[1:]), 0, 1)]  # find where d1 changes sign
        # this is the same length as d2 so [1, 1, -1, -1, 1] is [0, 0, 1, 0, 1]. Removing the first element centres it
        d1_indexes = (d1_indexes + np.roll(d1_indexes, -1)) != 0
        # now [1, 1, -1, -1, 1] is [0, 1, 1, 1, 1], so every index next to a change is included (both sides)
        d2 = np.gradient(d1)
        # d2_indexes = np.where(d2 > 0, 1, 0)  # if the point where d1 is 0 is a maximum in the line
        d2_indexes = d2 < 0  # find where there is a local maximum in the amplitudes
        indexes = np.logical_and(d1_indexes, d2_indexes)  # if an index satisfies both gradient conditions

        # a connected group is a run of True
        # candidate indexes for a peak are found by finding the edges of 1 index out of the connected groups
        options = np.argwhere([0, *np.logical_and(indexes[:-1], indexes[1:])]).flatten()
        diffs = np.ediff1d(options)  # find the differences between the positions of the candidate indexes
        # todo positions should be used so diffs = np.ediff1d(positions[options]) when f isn't equally spaced
        # find the 2 consecutive differences that add to the highest
        sums = (diffs + np.roll(diffs, -1))[:-1]  # ignore the final sum as it adds around the loop to the start!
        # chosen = options[np.argwhere(sums == np.amax(sums)).flatten()[0] + 1]  # select the run that is the biggest
        area_bounds = (options[np.argwhere(sums == np.amax(sums)).flatten()[0]],
                       options[np.argwhere(sums == np.amax(sums)).flatten()[0] + 2])
        # find the index of the maximum value within the chosen area
        i = np.argwhere(a == np.nanmax(a[np.arange(area_bounds[1] - area_bounds[0]) + area_bounds[0]])).flatten()
        # todo interpolate for frequency where d1 = 0 exactly?

        likely_correct = True  # todo replace with some condition that means it has to be a peak and not just noise!
        # confidence = peak_width? that's a good condition?

        if likely_correct:
            found = True
            print(f"Peak is f = {data[i, 0]} Hz")
        else:
            a[i] = np.nan  # make sure this frequency is ignored

    if found:
        plt.plot(data[:, 0], data[:, 1])
        plt.plot([data[i, 0], data[i, 0]], [min(data[:, 1]), max(data[:, 1])])
        plt.show()
    else:
        print("can't find a peak!")  # shouldn't need to be used in this version!
