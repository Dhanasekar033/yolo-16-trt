import sys
import time
import serial
from serial.tools import list_ports
# 0 = RED, 1 = GREEN, 2 = REVERSE, 3 = CONVEYOR
# def detect_port():
#     for port in list_ports.comports():
#         if "USB" in port.description or "Serial" in port.description:
#             return port.device
#     return None

def detect_port():
    ports = list_ports.comports()
    for port in ports:
        print(f"Found: {port.device} | {port.description} | hwid={port.hwid}")
        # Match CH340 by description, hwid, VID:PID (1a86:7523), or ttyUSB device path
        hwid = port.hwid.upper() if port.hwid else ""
        desc = port.description.upper() if port.description else ""
        if ("CH340" in desc or "CH340" in hwid or
                "1A86:7523" in hwid or
                "USB" in desc or
                port.device.startswith("/dev/ttyUSB")):
            return port.device
    return None

def calc_crc(data):
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc

def send_relay(ser, relay, state, device_addr=0xFF):
    value = 0xFF00 if state else 0x0000
    msg = [device_addr, 0x05, (relay >> 8) & 0xFF, relay & 0xFF, (value >> 8) & 0xFF, value & 0xFF]
    crc = calc_crc(msg)
    msg += [crc & 0xFF, (crc >> 8) & 0xFF]
    cmd = bytearray(msg)
    print(f"[SEND] relay={relay} {'ON' if state else 'OFF'}: {' '.join(f'{b:02X}' for b in cmd)}")
    ser.write(cmd)
    time.sleep(0.15)
    resp = ser.read(100)
    if resp:
        print(f"[RESP] {' '.join(f'{b:02X}' for b in resp)}")

if len(sys.argv) < 2:
    print("Usage: python3 test-relay.py <relay_number> [duration]")
    print("  relay: 0=RED, 1=GREEN, 2=REVERSE, 3=CONVEYOR")
    sys.exit(1)

relay = int(sys.argv[1])
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

port = detect_port()
if not port:
    print("No serial port found. Exiting.")
    sys.exit(1)

print(f"Port: {port} | Relay: {relay} | Duration: {duration}s")

ser = serial.Serial(port=port, baudrate=9600, parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=0.2)

send_relay(ser, relay, True)
time.sleep(duration)
send_relay(ser, relay, False)

ser.close()
print("Done.")