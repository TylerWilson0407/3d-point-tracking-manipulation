import serial
import time

ser = serial.Serial('COM3', 9600)

time.sleep(4)

ser.write('90')

time.sleep(4)

ser.close()
