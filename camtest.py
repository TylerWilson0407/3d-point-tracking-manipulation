import cv2
import numpy as np

# begin video capture
c1 = cv2.VideoCapture(0)
c2 = cv2.VideoCapture(1)
cv2.namedWindow('camera 1')
cv2.namedWindow('camera 2')
_,sizing_frame = c1.read()
cv2.moveWindow('camera 1',20,20)
cv2.moveWindow('camera 2',sizing_frame.shape[1]+18+20,20)

frame = [np.empty((sizing_frame.shape[0],sizing_frame.shape[1],3)),
         np.empty((sizing_frame.shape[0],sizing_frame.shape[1],3))]

while 1:
    _,frame[0] = c1.read()
    _,frame[1] = c2.read()

    cv2.imshow('camera 1',frame[0])
    cv2.imshow('camera 2',frame[1])

    if cv2.waitKey(5) == 27:
        break