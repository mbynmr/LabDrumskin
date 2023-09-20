import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.signal as sg
from tqdm import tqdm


def time_frequency_spectrum():
    data = np.loadtxt(r"C:\Users\mbynmr\OneDrive - The University of Nottingham" +
                           r"\Documents\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\py\raw.txt")
    times = data[:, 0]
    data_time = data[:, 1]
    time_end = np.max(times)  # s
    # data_rate = 1 / np.abs(times[1] - times[0])  # Hz
    number_of_samples = len(data_time)
    minimum_frequency = 1 / time_end
    maximum_frequency = (number_of_samples / 2) / time_end  # = data_rate / 2  # aka Nyquist frequency
    frequencies = np.linspace(start=minimum_frequency, stop=maximum_frequency, num=int((number_of_samples-1)/2),
                              endpoint=True)

    if number_of_samples % 2 == 0:
        frequencies_from_fft = np.array([0, *frequencies, maximum_frequency, *(-frequencies[::-1])])
    else:
        frequencies_from_fft = np.array([0, *frequencies, *(-frequencies[::-1])])

    data_frequency = np.fft.fft(data_time)
    # data_hilbert = sg.hilbert(data_frequency)
    # intensities = np.abs(data_hilbert)
    intensities = np.abs(data_frequency)


    intensities_to_plot = intensities[:int(len(intensities)/2)]
    plt.ion()

    ax = plt.subplots(num="fft of all time")[1]
    ax.plot(frequencies_from_fft[:int(len(intensities)/2)], intensities_to_plot, '-')

    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("power")
    plt.ylim([0, 100])
    plt.draw()
    plt.ioff()

def time_frequency_spectrum2electricboogaloo():
    data = np.loadtxt(r"C:\Users\mbynmr\OneDrive - The University of Nottingham" +
                           r"\Documents\Shared - Mechanical Vibrations of Ultrathin Films\Lab\data\py\raw.txt")
    times = data[:, 0]
    data_time = data[:, 1]
    time_end = np.max(times)  # s
    # data_rate = 1 / np.abs(times[1] - times[0])  # Hz
    number_of_samples = len(data_time)
    minimum_frequency = 1 / time_end
    maximum_frequency = (number_of_samples / 2) / time_end  # = data_rate / 2  # aka Nyquist frequency
    bins = int(5e3)
    # filter_type = 'top hat'
    filter_type = 'gauss'
    # filter_type = 'none'
    # end_frequency = maximum_frequency
    end_frequency = 1e4  # Hz


    frequencies = np.linspace(start=minimum_frequency, stop=maximum_frequency, num=int((number_of_samples-1)/2),
                              endpoint=True)
    # if number_of_samples % 2 == 0:
    #     frequencies_from_fft = np.array([0, *frequencies, maximum_frequency, *(-frequencies)])
    # else:
    #     frequencies_from_fft = np.array([0, *frequencies, *(-frequencies)])

    data_frequency = np.fft.fft(data_time)

    bin_centres = np.linspace(start=minimum_frequency, stop=end_frequency, num=bins, endpoint=False) + (
            (end_frequency - minimum_frequency) / bins) / 2
    bin_edges = np.linspace(start=minimum_frequency, stop=end_frequency, num=(bins + 1), endpoint=True)
    intensities = np.zeros([bins, number_of_samples])  # make 2D intensities array
    for i in tqdm(range(bins)):
        if filter_type == 'top hat':
            the_filter = (bin_edges[i] <= frequencies) * (frequencies < bin_edges[i + 1])
        elif filter_type == 'gauss':
            sigma = 100
            mean = bin_centres[i]
            # x = np.arange(start=-1, stop=1, step=4/data_frequency.shape[0])[1:]
            x = np.linspace(start=minimum_frequency, stop=maximum_frequency, num=int(data_frequency.shape[0]/2))[1:]
            the_filter = np.exp(-(x - mean) ** 2 * 0.5 * sigma ** -2)  # * (2 * np.pi * sigma ** 2) ** -2
        elif filter_type == 'none':
            the_filter = np.ones([int((data_frequency.shape[0] - 1) / 2)])
        else:
            raise ValueError(f"Invalid filter type '{filter_type}'")

        # Q3. a) i.
        if np.amax(the_filter) != 0:  # avoid div by zero error
            the_filter = the_filter / np.amax(the_filter)  # ensure the filter is normalised
        if data_frequency.shape[0] % 2 == 0:  # build the filter in frequency space
            data_frequency_filtered = data_frequency * np.array([0, *the_filter, 0, *the_filter[::-1]])
        else:
            data_frequency_filtered = data_frequency * np.array([0, *the_filter, *the_filter[::-1]])

        data_time_filtered = np.fft.ifft(data_frequency_filtered)
        data_time_filtered = np.real_if_close(data_time_filtered)
        # np.real_if_close fixes the rounding errors that make our time data complex
        # np.real_if_close will only output reals if the complex part is very small in magnitude

        # Q3. a) ii.
        data_hilbert = sg.hilbert(data_time_filtered)

        # Q3. a) iii.
        intensities[i, :] = np.abs(data_hilbert)

    # ---- choose a compression method (average or pick single)
    # compress = int(1e2)
    # intensities_to_compress = np.zeros([intensities.shape[0], int(intensities.shape[1] / compress), compress])
    # times_to_compress = np.zeros([int(times.shape[0] / compress), compress])
    # for j in tqdm(range(compress)):
    #     intensities_to_compress[:, :, j] = intensities[:, j::compress]
    #     times_to_compress[:, j] = times[j::compress]
    # intensities_to_plot = np.mean(intensities_to_compress, axis=2)
    # times_to_plot = np.mean(times_to_compress, axis=1)
    # ----
    intensities_to_plot = intensities
    times_to_plot = times
    # ----
    # graph = (np.array(range(times.shape[0])) % compress == 0)
    # intensities_to_plot = intensities[:, graph]
    # times_to_plot = times[graph]
    # ----

    # intensities_to_plot[bin_centres < 1.5e3, :] = 0  # quick and dirty remove sub 1kHz

    time_axis, frequency_axis = np.meshgrid(times_to_plot, bin_centres)
    ax = plt.subplots(subplot_kw={"projection": "3d"}, num="frequencies at times")[1]  # make a subplot that has 3D axes
    ax.plot_surface(time_axis, frequency_axis, intensities_to_plot, cmap=cm.jet)  # plot a surface on those axes
    # ax.plot_wireframe(time_axis, frequency_axis, intensities_to_plot)  # plot a surface on those axes
    ax.set_xlabel("time/s")
    ax.set_ylabel("frequency/Hz")
    ax.set_zlabel("power")
    plt.show()


def phase_information():
    # todo try to extract some phase information from the pulse...
    pass


def time_frequency_spectrum_original_coursework():
    time_end = 40  # s
    data_rate = 1200  # Hz
    number_of_samples = int(time_end * data_rate)
    minimum_frequency = 1 / time_end
    maximum_frequency = (number_of_samples / 2) / time_end  # = data_rate / 2  # aka Nyquist frequency
    bins = int(600)
    # filter_type = 'top hat'
    filter_type = 'gauss'
    # filter_type = 'none'
    # end_frequency = maximum_frequency
    end_frequency = 60  # Hz

    times = np.arange(start=0, stop=time_end, step=1 / data_rate)
    frequencies = np.linspace(start=minimum_frequency, stop=maximum_frequency, num=int((number_of_samples-1)/2),
                              endpoint=True)
    # if number_of_samples % 2 == 0:
    #     frequencies_from_fft = np.array([0, *frequencies, maximum_frequency, *(-frequencies)])
    # else:
    #     frequencies_from_fft = np.array([0, *frequencies, *(-frequencies)])

    data_time = np.loadtxt("Q3/Fourier_Filtering/signal.mat")
    if data_time.shape[0] != number_of_samples:
        raise ValueError(f"Unexpected length of imported data: {data_time.shape[0]} != {number_of_samples}")

    data_frequency = np.fft.fft(data_time)

    bin_centres = np.linspace(start=minimum_frequency, stop=end_frequency, num=bins, endpoint=False) + (
            (end_frequency - minimum_frequency) / bins) / 2
    bin_edges = np.linspace(start=minimum_frequency, stop=end_frequency, num=(bins + 1), endpoint=True)
    intensities = np.zeros([bins, number_of_samples])  # make 2D intensities array
    for i in tqdm(range(bins)):
        if filter_type == 'top hat':
            the_filter = (bin_edges[i] <= frequencies) * (frequencies < bin_edges[i + 1])
        elif filter_type == 'gauss':
            sigma = 1
            mean = bin_centres[i]
            # x = np.arange(start=-1, stop=1, step=4/data_frequency.shape[0])[1:]
            x = np.linspace(start=minimum_frequency, stop=maximum_frequency, num=int(data_frequency.shape[0]/2))[1:]
            the_filter = np.exp(-(x - mean) ** 2 * 0.5 * sigma ** -2)  # * (2 * np.pi * sigma ** 2) ** -2
        elif filter_type == 'none':
            the_filter = np.ones([int((data_frequency.shape[0] - 1) / 2)])
        else:
            raise ValueError(f"Invalid filter type '{filter_type}'")

        # Q3. a) i.
        if np.amax(the_filter) != 0:  # avoid div by zero error
            the_filter = the_filter / np.amax(the_filter)  # ensure the filter is normalised
        if data_frequency.shape[0] % 2 == 0:  # build the filter in frequency space
            data_frequency_filtered = data_frequency * np.array([0, *the_filter, 0, *the_filter[::-1]])
        else:
            data_frequency_filtered = data_frequency * np.array([0, *the_filter, *the_filter[::-1]])

        data_time_filtered = np.fft.ifft(data_frequency_filtered)
        data_time_filtered = np.real_if_close(data_time_filtered)
        # np.real_if_close fixes the rounding errors that make our time data complex
        # np.real_if_close will only output reals if the complex part is very small in magnitude

        # Q3. a) ii.
        data_hilbert = sg.hilbert(data_time_filtered)

        # Q3. a) iii.
        intensities[i, :] = np.abs(data_hilbert)

    # ---- choose a compression method (average or pick single)
    compress = int(300)
    intensities_to_compress = np.zeros([intensities.shape[0], int(intensities.shape[1] / compress), compress])
    times_to_compress = np.zeros([int(times.shape[0] / compress), compress])
    for j in tqdm(range(compress)):
        intensities_to_compress[:, :, j] = intensities[:, j::compress]
        times_to_compress[:, j] = times[j::compress]
    intensities_to_plot = np.mean(intensities_to_compress, axis=2)
    times_to_plot = np.mean(times_to_compress, axis=1)
    # ----
    # graph = (np.array(range(times.shape[0])) % compress == 0)
    # intensities_to_plot = intensities[:, graph]
    # times_to_plot = times[graph]
    # ----

    time_axis, frequency_axis = np.meshgrid(times_to_plot, bin_centres)
    ax = plt.subplots(subplot_kw={"projection": "3d"})[1]  # make a subplot that has 3D axes
    ax.plot_surface(time_axis, frequency_axis, intensities_to_plot, cmap=cm.jet)  # plot a surface on those axes
    ax.set_xlabel("time/s")
    ax.set_ylabel("frequency/Hz")
    ax.set_zlabel("power")
    plt.show()
