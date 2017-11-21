def moveArm(ser, angles):
    import serial
    import numpy as np
    import cv2

    angles += np.array([90, 0, 45, 0, 90, 90, 0])

##    a_max = np.array([90, 135, 135, 135, 135, 135, 45])
##    a_min = np.array([45, 90, 45, 45, 45, 45, 45])

    a_max = np.array([150, 180, 180, 180, 180, 180, 45])
    a_min = np.array([30, 90, 0, 0, 0, 0, 45])

##    print angles

    angles = np.around(angles).astype(int)
    angles = np.minimum(angles, a_max)
    angles = np.maximum(angles, a_min)

    print angles

##    print angles

    for i in range(7):
        ser.write(chr(int(i)))
        check = ser.read()
        ser.write(chr(int(angles[i])))
        check = ser.read()
