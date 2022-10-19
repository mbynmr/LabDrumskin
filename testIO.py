import time

import nidaqmx
from pymeasure.instruments.agilent import Agilent33521A
import pyvisa
import binascii


def test():
    # Agilent33521A has code USB0::0x0957::0x1607::MY50003212::INSTR
    a335_resource = 'USB0::0x0957::0x1607::MY50003212::INSTR'
    # RSDG 805 has code USB0::0xF4ED::0xEE3A::SDG08CBQ5R1442::INSTR
    rs805_resource = 'USB0::0xF4ED::0xEE3A::SDG08CBQ5R1442::INSTR'

    rm = pyvisa.ResourceManager()

    a335 = rm.open_resource(a335_resource)
    # rs805 = rm.open_resource(rs805_resource)
    print(a335.query('*IDN?'))
    # print(rs805.query('*IDN?'))
    # print(rs805.query('IQ:AMPLitude?'))
    # rs805.write('IQ:AMPL 10')
    # print(rs805.query('IQ:AMPLitude?'))

    # print(a335.query('VOLTage:UNIT?'))
    a335.write('DISP:TEXT "setting up..."')

    a335.write('OUTPut:ON')
    a335.write('APPLy:PULSe 1.666667, MAX')

    a335.write('DISP ON')

    a335.write('FUNCtion:PULSe:WIDTh 0.0001')
    time.sleep(2)
    a335.write('FUNCtion:PULSe:WIDTh 0.00001')
    time.sleep(2)
    a335.write('FUNCtion:PULSe:WIDTh 0.0001')
    # a335.write('AMPL 10')
    # print(a335.query('AMPLitude?'))
