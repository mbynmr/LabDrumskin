import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import datetime

# from IO_setup import set_up_signal_generator_pulse
from my_tools import normalise


def play_mic_signal(array=None, t=None):
    # plays the inputted array, or 'outputs/raw.txt' if array is None
    sd.default.samplerate = 20000.0  # default is 44100 or 48000

    if array is None:  # no inputted array, read raw.txt
        array = np.loadtxt("outputs/raw.txt")
        t = array[:, 0]
        v = array[:, 1]
    else:  # inputted array
        array = np.array(array)
        if array.ndim == 1:
            if t is None:
                t = [0, 0.2]  # no time info anywhere, assume 0.2s long.
            elif len(t) == 1:
                t = [0, t]
            elif max(t) == min(t):  # wtf?
                t = [0, 0.2]
            v = array
        else:  # simple, the input contained t and v
            t = array[:, 0]
            v = array[:, 1]

    samplerate = v.shape[0] / (max(t) - min(t))
    sd.play(normalise(v), samplerate=samplerate, blocking=False, loop=True)
    print("audio playing... raw mic signal")
    '''
    blocking (bool, optional) –
    If False (the default), return immediately (but playback continues in the background),
    if True, wait until playback is finished.
    A non-blocking invocation can be stopped with sd.stop() or turned into a blocking one with sd.wait().
    '''

    # time.sleep(2)
    # array = np.loadtxt("outputs/audio/raw pulse perfect.txt")
    # t = array[:, 0]
    # v = array[:, 1]
    # samplerate = t.shape[0] / (t.max() - t.min())
    # v = (v + v.min() if v.min() >= 0 else v - v.min())
    # v = (2 * v / v.max()) - 1
    # sd.play(v, samplerate=samplerate, blocking=False, loop=True)
    # time.sleep(2)
    # array = np.loadtxt("outputs/audio/raw pulse.txt")
    # t = array[:, 0]
    # v = array[:, 1]
    # samplerate = t.shape[0] / (t.max() - t.min())
    # v = (v + v.min() if v.min() >= 0 else v - v.min())
    # v = (2 * v / v.max()) - 1
    # sd.play(v, samplerate=samplerate, blocking=False, loop=True)
    # time.sleep(2)
    # sd.stop()
    # # print("audio stopped")


def play_spectra(file, t=0.2):
    try:
        s = np.array(file.split('/')[-1].split('_')[:4], dtype=np.int_)
        if (file.split('_')[6][0] != 'P' and file.split('_')[6][0:2] != 'TP') or datetime.datetime(
                year=s[0], month=s[1], day=s[2], hour=s[3]).timestamp() < datetime.datetime(
                year=2024, month=4, day=17, hour=11).timestamp():
            raise ValueError
    except ValueError:
        print("unable to play spectra - file name indicates it is not a pulse spectra saved after 2024/04/16")
        # 2024_04_17_13_40_01_P33_PSY1_1_20.txt
        # fft(signal) gives:
        # [0 + 0i, 1 + 1j, 2 + 2j, 3 + 3j, 100 + 0j, 3 + 3j, 2 + 2j, 1 + 1j]
        # I've been saving [1 + 1j, 2 + 2j, 3 + 3j] in r and i which misses out the 100 in r.
        # 17/04/2024 this has been CORRECTED. Also frequencies have been corrected as they were 1 too short.

        # the below used to be true:
        # fft = np.fft.fft(response - np.mean(response))
        # # np.allclose(np.real(fft[1:int(len(fft)/2)]), np.real(fft[int(len(fft)/2)+1:][::-1])) == True
        # # np.allclose(np.imag(fft[1:int(len(fft)/2)]), -np.imag(fft[int(len(fft)/2)+1:][::-1])) == True
        # data = np.abs(fft[1:int(num / 2)])  # legacy oh noooooooooooo
        # datar = np.real(fft[1:int(num / 2)])  # real
        # datai = np.imag(fft[1:int(num / 2)])  # imaginary

        # but now it is not necessarily true. We need that middle value and idk if it messes up that True
        # https://www.dsprelated.com/showarticle/901.php
        return

    array = np.loadtxt(file)
    datar = array[:, 2]
    datai = array[:, 3]
    data = datar * 0.01 + (999 * datai * 1j)  # just here for type!! not actually how they go together.

    a = np.zeros(len(data) * 2, dtype=data.dtype)
    # UNEQUAL!!!!! remember that the extra entry in imag is useless, but in real is not.
    # [0 + 0i, 1 + 1j, 2 + 2j, 3 + 3j, 100 + 0j, 3 + 3j, 2 + 2j, 1 + 1j]
    a[1:len(data) + 1] = datar
    a[1:len(data)] = a[1:len(data)] + (datai * 1j)[:-1]
    a[len(data) + 1:] = datar[:-1][::-1]
    a[len(data) + 1:] = a[len(data) + 1:] - (datai * 1j)[:-1][::-1]
    ifft = np.fft.ifft(a)
    v = np.real_if_close(ifft)
    # v = np.real(ifft)

    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(v)
    plt.ioff()

    samplerate = 20000.0  # Hz
    sd.play(normalise(v), samplerate=samplerate, blocking=False, loop=True)
    print("audio playing... spectra")


def play_stop():
    sd.stop()
    print("audio stopped")


def notify_finish():
    # path = r'C:\Windows\Media\Windows Pop-up Blocked.wav'  # this seemed like a good system file to use 🤣
    # path = r'C:\Windows\Media\tada.wav'
    path = r'C:\Windows\Media\Speech On.wav'
    samples, samplerate = sf.read(path)
    sd.play(samples, samplerate=samplerate, blocking=False, loop=False)
    
    # sig_gen = set_up_signal_generator_pulse()
    #
    # my_WFM = 262000 * [1]  # works fine (it is just below 1MB)
    # my_WFM = 263000*[1]  # fails (it is just above 1MB)
    # # my_WFM = np.sin(2 * np.pi * np.linspace(0, 20, 1000))
    # print(sig_gen.query('*IDN?'))  # works fine
    #
    # # sig_gen.timeout = 300000
    # sig_gen.write('*CLS;*RST')
    # _ = sig_gen.query('*OPC?')
    # sig_gen.write('SOURce1:DATA:VOLatile:CLEar')
    # # sig_gen.write("FREQ 1000")  # frequency is 1kHz
    # # sig_gen.write("VOLT 1")  # amplitude is 1Vpp
    # sig_gen.write('FORM:BORD NORM')  # set the byte order
    #
    # print('Downloading Waveform...')
    # bytes_sent = sig_gen.write_binary_values('SOUR1:DATA:ARB myARB,', my_WFM, datatype='f', is_big_endian=True)
    # print(bytes_sent)
    # sig_gen.write('*WAI')  # Wait for the waveform to load
    # print('Download Complete')
    #
    # print("OUTPUT ON NOW PLZ TY")
    # sig_gen.write("OUTPut ON")
    #
    # # sig_gen.close()  # close connection to device
