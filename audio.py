import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time


def play_mic_signal(array=None, t=0.2):
    sd.default.samplerate = 20000.0  # default is 44100 or 48000 #todo find if i can use this gd
    if array is None:
        array = np.loadtxt("outputs/raw.txt")
    t = array[:, 0]
    v = array[:, 1]

    samplerate = t.shape[0] / (t.max() - t.min())
    v = (v + v.min() if v.min() >= 0 else v - v.min())
    v = (2 * v / v.max()) - 1
    sd.play(v, samplerate=samplerate, blocking=False, loop=True)
    print("audio playing... raw mic signal")
    '''
    blocking (bool, optional) –
    If False (the default), return immediately (but playback continues in the background),
    if True, wait until playback is finished.
    A non-blocking invocation can be stopped with sd.stop() or turned into a blocking one with sd.wait().
    '''
    time.sleep(2)

    array = np.loadtxt("outputs/audio/raw pulse perfect.txt")
    t = array[:, 0]
    v = array[:, 1]
    samplerate = t.shape[0] / (t.max() - t.min())
    v = (v + v.min() if v.min() >= 0 else v - v.min())
    v = (2 * v / v.max()) - 1
    sd.play(v, samplerate=samplerate, blocking=False, loop=True)
    time.sleep(2)

    array = np.loadtxt("outputs/audio/raw pulse.txt")
    t = array[:, 0]
    v = array[:, 1]
    samplerate = t.shape[0] / (t.max() - t.min())
    v = (v + v.min() if v.min() >= 0 else v - v.min())
    v = (2 * v / v.max()) - 1
    sd.play(v, samplerate=samplerate, blocking=False, loop=True)
    time.sleep(2)
    sd.stop()
    # print("audio stopped")


def play_spectra(file, t=0.2):
    array = np.loadtxt(file)
    datar = array[:, 2]
    datai = array[:, 3]
    data = datar + (datai * 1j)  # just here for type!! not actually how they go together.

    # fft = np.fft.fft(response - np.mean(response))
    # # np.allclose(np.real(fft[1:int(len(fft)/2)]), np.real(fft[int(len(fft)/2)+1:][::-1])) == True
    # # np.allclose(np.imag(fft[1:int(len(fft)/2)]), -np.imag(fft[int(len(fft)/2)+1:][::-1])) == True
    # data = np.abs(fft[1:int(num / 2)])  # legacy oh noooooooooooo
    # datar = np.real(fft[1:int(num / 2)])  # real
    # datai = np.imag(fft[1:int(num / 2)])  # imaginary
    # https://www.dsprelated.com/showarticle/901.php
    #
    # fft(signal) gives:
    # [0 + 0i, 1 + 1j, 2 + 2j, 3 + 3j, 9 + 0j, 3 + 3j, 2 + 2j, 1 + 1j]
    # I've been saving [1 + 1j, 2 + 2j, 3 + 3j] in r and i which misses out the 9 in r.


    a = np.zeros(len(data) * 2 + 2, dtype=data.dtype)
    a[1:len(data) + 1] = datar + (datai * 1j)
    a[len(data) + 2:] = datar[::-1] - (datai[::-1] * 1j)
    ifft = np.fft.ifft(a)
    v = np.real(ifft)
    fig, ax = plt.subplots()
    ax.plot(v)
    ax.plot(np.imag(ifft))
    samplerate = 20000.0
    v = (v + v.min() if v.min() >= 0 else v - v.min())
    v = (2 * v / v.max()) - 1
    sd.play(v, samplerate=samplerate, blocking=False, loop=True)
    print("audio playing... spectra")


def play_stop():
    sd.stop()
    print("audio stopped")
