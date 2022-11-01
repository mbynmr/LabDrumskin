import pyvisa

from IO_test import waveform_generators, picoscope, picoscope_check


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
    # todo find optimal pulse width
    # maybe try different waveform generators
    a335.write('FUNCtion:PULSe:WIDTh 0.01')

    # a335.write('DISP:TEXT "Test running! Be quiet please"')

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

    # a335.write('OUTPut OFF')
    # a335.write('OUTPut ON')
    a335.write('APPLy:SINusoid 50, 10')  # todo

    # a335.write('DISP:TEXT "Test running! Be quiet please"')

    return a335
    # return rs805
