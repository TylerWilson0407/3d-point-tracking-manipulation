import serial
import numpy as np
import cv2
import time
connected = False
ser = serial.Serial('COM3', 9600)
##ser.open()

while not connected:
    serin = ser.read()
    connected = True

angles1 = np.array([80, 135, 45, 45, 45, 45, 45])
angles2 = np.array([135, 135, 45, 135, 135, 180, 45])

time.sleep(1)

for i in range(7):
    ser.write(chr(int(i)))
    check = ser.read()
    ser.write(chr(int(angles1[i])))
    check = ser.read()

time.sleep(2)

for i in range(7):
    ser.write(chr(int(i)))
    check = ser.read()
    ser.write(chr(int(angles2[i])))
    check = ser.read()

time.sleep(2)

ser.close()
