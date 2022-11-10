# import nidaqmx
# from pymeasure.instruments.agilent import Agilent33521A
import pyvisa
# import binascii
import time


def waveform_generators():
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


def picoscope_check():
    from picosdk.discover import find_all_units

    scopes = find_all_units()

    for scope in scopes:
        print(scope.info)
        scope.close()

    # the above outputs the same as the below... c
    from picosdk.ps2000 import ps2000

    with ps2000.open_unit() as device:
        print(f'{device.info}')


def picoscope():
    print()


def picoscope_bad():
    # signal generator time >:D
    import ctypes
    from picosdk.functions import assert_pico_ok
    from picosdk.ps2000a import ps2000a as ps  # todo check I don't need to be 2205a or whateverx

    status = {}
    chandle = ctypes.c_int16()
    status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(chandle), None)

    try:
        assert_pico_ok(status["openunit"])
    except:
        # powerstate becomes the status number of openunit
        powerstate = status["openunit"]

        # If powerstate is the same as 282 then it will run this if statement
        if powerstate == 282:
            # Changes the power input to "PICO_POWER_SUPPLY_NOT_CONNECTED"
            status["ChangePowerSource"] = ps.ps2000aChangePowerSource(chandle, 282)
        # If the powerstate is the same as 286 then it will run this if statement
        elif powerstate == 286:
            # Changes the power input to "PICO_USB3_0_DEVICE_NON_USB3_0_PORT"
            status["ChangePowerSource"] = ps.ps2000aChangePowerSource(chandle, 286)
        else:
            print("fek")
            raise  # todo this is horrible still

    # Output a sine wave with peak-to-peak voltage of 2 V and frequency of 5 kHz
    # handle = chandle
    # offsetVoltage = 0
    # pkToPk = 2000000
    # waveType = ctypes.c_int16(0) = PS2000A_SINE
    # startFrequency = 10 kHz
    # stopFrequency = 10 kHz
    # increment = 0
    # dwellTime = 1
    # sweepType = ctypes.c_int16(1) = PS2000A_UP
    # operation = 0
    # shots = 0
    # sweeps = 0
    # triggerType = ctypes.c_int16(0) = PS2000A_SIGGEN_RISING
    # triggerSource = ctypes.c_int16(0) = PS2000A_SIGGEN_NONE
    # extInThreshold = 1
    wavetype = ctypes.c_int16(0)
    sweepType = ctypes.c_int32(0)
    triggertype = ctypes.c_int32(0)
    triggerSource = ctypes.c_int32(0)

    Vpp = 10e6  # peak to peak voltage in micro V so 2,000,000 = 2V (just stick e6 after the voltage you want)
    freqs = [5e3, 5e3]  # start, stop freqs in Hz

    status["SetSigGenBuiltIn"] = ps.ps2000aSetSigGenBuiltIn(chandle, 0, Vpp, wavetype, freqs[0], freqs[1], 0, 1,
                                                            sweepType, 0, 0, 0, triggertype, triggerSource, 1)
    time.sleep(5)
    assert_pico_ok(status["SetSigGenBuiltIn"])
    status["close"] = ps.ps2000aCloseUnit(chandle)
    assert_pico_ok(status["close"])

    # Displays the status returns
    print(status)
