import pyvisa
import nidaqmx as ni
import numpy as np
import matplotlib.pyplot as plt
import time

from my_tools import ax_lims, temp_get


def set_up_daq(mode, c1, c2=None, rate=int(20e3), t=0.2):
    # easy creation of one of the two task types needed
    task = ni.Task()
    match mode:
        case 'dual':
            num = int(np.ceil(int(rate / 2) * t))  # number of samples to measure
            task.ai_channels.add_ai_voltage_chan(c1, min_val=-10.0, max_val=10.0)
            task.timing.cfg_samp_clk_timing(rate=int(rate / 2), samps_per_chan=num)
            task.ai_channels.add_ai_voltage_chan(c2, min_val=-10.0, max_val=10.0)
        case 'single':
            num = int(np.ceil(rate * t))  # number of samples to measure
            task.ai_channels.add_ai_voltage_chan(c1, min_val=-10.0, max_val=10.0)
            task.timing.cfg_samp_clk_timing(rate=rate, samps_per_chan=num)
        case _:
            raise ValueError("'dual' or 'single' mode for set_up_daq_task")
    return task, num


def grab_temp(dev, chan, num=1000):
    with ni.Task() as task:
        task.ai_channels.add_ai_voltage_chan(dev + '/' + chan, min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=10000, samps_per_chan=10000)
        return np.mean(temp_get(task.read(num)))


def list_devices():
    # Lists all connected devices to the system, and their channels for format 'Dev1/ai0'
    return [d.name for d in ni.system.System.local().devices], ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7']


def set_up_signal_generator_pulse():
    a335_resource = 'USB0::0x0957::0x1607::MY50003212::INSTR'
    # RSDG 805 has code USB0::0xF4ED::0xEE3A::SDG08CBQ5R1442::INSTR
    rs805_resource = 'USB0::0xF4ED::0xEE3A::SDG08CBQ5R1442::INSTR'

    rm = pyvisa.ResourceManager()

    a335 = rm.open_resource(a335_resource)
    # rs805 = rm.open_resource(rs805_resource)

    # a335.write('DISP:TEXT "setting up..."')

    # a335.write('OUTPut:ON')
    # a335.write('APPLy:PULSe 1.5, MAX')
    a335.write('APPLy:PULSe 1.666667, MAX')
    # 100 micro secs = 0.0001
    # 1 micro sec = 0.00001
    # below 100 micro secs the piezo makes less of a sound
    # to do find optimal pulse width
    # maybe try different waveform generators
    a335.write('FUNCtion:PULSe:WIDTh 0.0001')  # best compromise between width and power is 0.0001 = 100 micro second

    # a335.write('DISP:TEXT "Test running! Be quiet please"')

    return a335
    # return rs805


def set_up_signal_generator_sine():
    a335_resource = 'USB0::0x0957::0x1607::MY50003212::INSTR'
    a335 = pyvisa.ResourceManager().open_resource(a335_resource)
    a335.write('OUTPut ON')
    a335.write('APPLy:SINusoid 50, 10')
    return a335


def calibrate(**kwargs):
    task, num = set_up_daq(**kwargs)
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([0, 0], [0, 0], 'k')
    axt = ax.twinx()
    linet, = axt.plot([0, 0], [0, 0], 'r')
    # plt.xlabel("elapsed time / s")
    temps = []
    realts = []
    starttime = time.time()
    while plt.fignum_exists(fig.number):
        _, t = task.read(num)
        temps.append(np.mean(t))
        realt = np.mean(temp_get(t))
        realts.append(realt)
        plt.title(f"reading is {np.mean(t):.6g}V, meaning temp is {realt:.6g}C")
        # print(np.mean(t))

        elapsed = time.time() - starttime
        line.set_xdata(np.linspace(start=0, stop=elapsed, endpoint=True, num=len(temps)))
        line.set_ydata(temps)
        ax.set_xlim([0, elapsed])
        ax.set_ylim(ax_lims(temps))

        linet.set_xdata(np.linspace(start=0, stop=elapsed, endpoint=True, num=len(temps)))
        linet.set_ydata(realts)
        axt.set_xlim([0, elapsed])
        axt.set_ylim(ax_lims(realts))

        fig.canvas.draw()
        fig.canvas.flush_events()
