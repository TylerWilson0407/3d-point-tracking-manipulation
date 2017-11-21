import cv2
import numpy as np
import serial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time

from pointTrack import pointTrack
import TriangulatePoint
import Plot3Dpoint
import Kalman
from moveArm import moveArm
import Kinematics3 as km

# begin video capture
c1 = cv2.VideoCapture(0)
c2 = cv2.VideoCapture(1)
cv2.namedWindow('camera 1')
cv2.namedWindow('camera 2')
_,sizing_frame = c1.read()
cv2.moveWindow('camera 1',20,20)
cv2.moveWindow('camera 2',sizing_frame.shape[1]+18+20,20)

# initialize 3D plotting
app, w, SP, LPh, LPr, armLines, jointPos, jointCoords = Plot3Dpoint.Initialize()

# initialize kinematic chain
KC = km.InitializeKinematicChain2()

# target points
world3D = np.array([[50, 50, 50],
           [50, 50, 100],
           [100,  50, 50]])

theta = np.array([45, -90, 45, 0, 0, 135], dtype = np.float)

centroid = world3D[:,1]

headingvec = np.cross(world3D[:,0]-world3D[:,1],world3D[:,0]-world3D[:,2])
if np.linalg.norm(headingvec) != 0:
    headingvec /= np.linalg.norm(headingvec)
headingvec_scaled = 75*headingvec
##headingvec_scaled = headingvec
headingvec_scaled += centroid

rollvec = world3D[:,2]-world3D[:,1]
if np.linalg.norm(rollvec) != 0:
    rollvec /= np.linalg.norm(rollvec)
rollvec_scaled = 75*rollvec
##rollvec_scaled = rollvec
rollvec_scaled += centroid

pos = np.empty((3,3))

pos[:3,:] = world3D.T

Xg = np.empty((6))
Xg[:3] = centroid
Xg[3:6] = headingvec

p = km.FKsolve(KC, theta)
arm_pos = p[:,:3,3]

np.set_printoptions(suppress=True,precision=2)
print p[5,:3,3]
print p[5,:3,:3]

frame = [np.empty((sizing_frame.shape[0],sizing_frame.shape[1],3)),
         np.empty((sizing_frame.shape[0],sizing_frame.shape[1],3))]

##Plot3Dpoint.HandUpdate(SP,LPh,LPr,pos,np.array([centroid,headingvec_scaled]),
##                       np.array([centroid,rollvec_scaled]))
##
##Plot3Dpoint.ArmUpdate(armLines, jointPos, jointCoords, arm_pos,p)
##
##theta, arm_pos, p = km.IKsolveDLS(KC, theta, Xg)


while 1:
    _,frame[0] = c1.read()
    _,frame[1] = c2.read()

    cv2.imshow('camera 1',frame[0])
    cv2.imshow('camera 2',frame[1])

    theta, arm_pos, p = km.IKsolve3(KC, theta, Xg)

    Plot3Dpoint.HandUpdate(SP,LPh,LPr,pos,np.array([centroid,headingvec_scaled]),
                           np.array([centroid,rollvec_scaled]))

    Plot3Dpoint.ArmUpdate(armLines, jointPos, jointCoords, arm_pos,p)

    if cv2.waitKey(5)==27:
        break

cv2.destroyAllWindows()
c1.release()
c2.release()
Plot3Dpoint.close(w)
