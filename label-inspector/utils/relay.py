"""Modbus-RTU relay board control (CH340 USB serial).

Same wire protocol as relay_test.py — function 0x05 "write single coil" with
a CRC-16/MODBUS tail — wrapped in a small class so run.py can hold the port
open for the whole session instead of re-opening it per pulse.

Relay map (as labelled on the machine):
    0 = RED / winding machine start   1 = GREEN
    2 = REVERSE                       3 = CONVEYOR
"""

import time

import serial
from serial.tools import list_ports


def detect_port():
    """Find the CH340 USB-serial adapter the relay board sits behind."""
    for port in list_ports.comports():
        hwid = port.hwid.upper() if port.hwid else ""
        desc = port.description.upper() if port.description else ""
        if ("CH340" in desc or "CH340" in hwid or
                "1A86:7523" in hwid or
                "USB" in desc or
                port.device.startswith("/dev/ttyUSB")):
            return port.device
    return None


def calc_crc(data):
    """CRC-16/MODBUS over the message bytes."""
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


class RelayController:
    """Open the relay board once and drive individual coils.

    `enabled=False` (or no port found when `required=False`) turns every call
    into a no-op print, so the vision side can still be run on a desk with no
    relay board attached.
    """

    def __init__(self, port=None, baudrate=9600, addr=0xFF, enabled=True,
                 required=False, verbose=False):
        self.addr = addr
        self.verbose = verbose
        self.ser = None
        self.port = None
        self.states = {}

        if not enabled:
            print("[relay] disabled (--no-relay)")
            return

        self.port = port or detect_port()
        if not self.port:
            msg = "no CH340 serial port found"
            if required:
                raise RuntimeError(f"[relay] {msg}")
            print(f"[relay] {msg} — running without relay control")
            return

        self.ser = serial.Serial(port=self.port, baudrate=baudrate,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 bytesize=serial.EIGHTBITS, timeout=0.2)
        print(f"[relay] connected on {self.port}")

    @property
    def active(self):
        return self.ser is not None

    def set(self, relay, state):
        """Switch one coil on/off. Silently ignored when no board is attached."""
        self.states[relay] = bool(state)
        if not self.active:
            return False

        value = 0xFF00 if state else 0x0000
        msg = [self.addr, 0x05, (relay >> 8) & 0xFF, relay & 0xFF,
               (value >> 8) & 0xFF, value & 0xFF]
        crc = calc_crc(msg)
        msg += [crc & 0xFF, (crc >> 8) & 0xFF]
        cmd = bytearray(msg)

        if self.verbose:
            print(f"[relay] {relay} {'ON' if state else 'OFF'}: "
                  f"{' '.join(f'{b:02X}' for b in cmd)}")
        self.ser.write(cmd)
        time.sleep(0.15)
        resp = self.ser.read(100)
        if resp and self.verbose:
            print(f"[relay] resp: {' '.join(f'{b:02X}' for b in resp)}")
        return True

    def on(self, relay):
        return self.set(relay, True)

    def off(self, relay):
        return self.set(relay, False)

    def pulse(self, relay, duration=2.0):
        self.set(relay, True)
        time.sleep(duration)
        self.set(relay, False)

    def all_off(self):
        for relay in list(self.states):
            if self.states[relay]:
                self.set(relay, False)

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None
