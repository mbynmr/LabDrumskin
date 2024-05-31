import pyvisa
import nidaqmx as ni


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
