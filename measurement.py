import nidaqmx
import numpy as np
import pyvisa
import time
import matplotlib.pyplot as plt

from my_tools import round_sig_figs


def measure(freq=None, freqstep=5, t=2):
    """Measures a ]piezo.
    freq=[minf, maxf] is the minimim and maximum frequencies to sweep between
    freqstep=df is the frequency step between measurements
    t=t is the length of time a measurement takes"""
    # Agilent33521A has code USB0::0x0957::0x1607::MY50003212::INSTR
    if freq is None:
        freq = [50, 2000]

    sig_gen = set_up_signal_generator_sine()

    plt.ion()
    fig, ax = plt.subplots()
    plt.ylabel("Amplitude / V")
    plt.xlabel("time / s")

    x = np.arange(1000)
    line, = ax.plot(x, np.zeros_like(x))

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(t)

    with nidaqmx.Task() as task:
        with open("outputs/output.txt", 'w') as out:
            task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
            for f in np.linspace(freq[0], freq[1], int((1 + freq[1] - freq[0]) / freqstep)):
                # a335.write(f'{f}')

                signal = task.read(number_of_samples_per_channel=1000)
                # y = np.real_if_close(np.fft.fft(signal))
                y = signal

                sig_gen.write(f'APPLy:SINusoid {f}, 10')

                # update figure
                line.set_ydata(y)
                ax.set_ylim([round_sig_figs(min(y) * 1.05, 2), round_sig_figs(max(y) * 1.05, 2)])
                fig.canvas.draw()
                fig.canvas.flush_events()

                out.write(f"{f:.6g} {np.mean(signal):.6g}\n")

                # time.sleep(1)


def set_up_signal_generator_pulse():
    a335_resource = 'USB0::0x0957::0x1607::MY50003212::INSTR'
    # RSDG 805 has code USB0::0xF4ED::0xEE3A::SDG08CBQ5R1442::INSTR
    rs805_resource = 'USB0::0xF4ED::0xEE3A::SDG08CBQ5R1442::INSTR'

    rm = pyvisa.ResourceManager()

    a335 = rm.open_resource(a335_resource)
    # rs805 = rm.open_resource(rs805_resource)

    a335.write('DISP:TEXT "setting up..."')

    a335.write('OUTPut:ON')
    a335.write('APPLy:PULSe 1.666667, MAX')
    # 100 micro secs = 0.0001
    # 1 micro sec = 0.00001
    # below 100 micro secs the piezo makes less of a sound
    a335.write('FUNCtion:PULSe:WIDTh 0.0001')

    a335.write('DISP:TEXT "Test running! Be quiet please"')

    return a335
    # return rs805


def set_up_signal_generator_sine():
    a335_resource = 'USB0::0x0957::0x1607::MY50003212::INSTR'
    # RSDG 805 has code USB0::0xF4ED::0xEE3A::SDG08CBQ5R1442::INSTR
    rs805_resource = 'USB0::0xF4ED::0xEE3A::SDG08CBQ5R1442::INSTR'

    rm = pyvisa.ResourceManager()

    a335 = rm.open_resource(a335_resource)
    # rs805 = rm.open_resource(rs805_resource)

    # a335.write('DISP:TEXT "setting up..."')

    a335.write('OUTPut:ON')
    a335.write('APPLy:SINusoid 50, 10')

    # a335.write('DISP:TEXT "Test running! Be quiet please"')

    return a335
    # return rs805
