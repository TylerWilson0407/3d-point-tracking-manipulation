import numpy as np
import serial
import time
import cv2
from moveArm import moveArm
from IKsolve import IKsolve

connected = False
ser = serial.Serial('COM3', 9600)

while not connected:
    serin = ser.read()
    connected = True

theta = np.array([80, 135, 45, 45, 45, 45, 45])
t_waitmove = 0.5
t_moved = time.time()

Xg = np.array([140, 0, 150,0,1,0])

theta_arm = IKsolve(theta[:6], Xg)
theta[:6] = theta_arm

print theta

moveArm(ser,theta)

while 1:

    theta_arm = IKsolve(theta[:6], Xg)
    theta[:6] = theta_arm

    time.sleep(1)
    print theta
    
    # press ESC to break loop
    if cv2.waitKey(5)==27:
        break
