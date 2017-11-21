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
from IKsolve import IKsolve

# calibration

num_points = 3 # number of points tracked(per camera)
num_cams = 2 # number of cameras

stereo_calib_file = open('Calibration Data\stereocalib.pkl', 'rb')
world_frame_file = open('Calibration Data\worldtransformmatrix.pkl', 'rb')
kalman_calib_file = open('Calibration Data\kalmancalib.pkl', 'rb')
pink_hsv_file = open('Calibration Data\HSVpink_1.pkl', 'rb')
blue_hsv_file = open('Calibration Data\HSVblue_1.pkl', 'rb')
green_hsv_file = open('Calibration Data\HSVgreen_1.pkl', 'rb')

StereoCalib = pickle.load(stereo_calib_file)
KalmanCalib = pickle.load(kalman_calib_file)
TransformMat = pickle.load(world_frame_file)
calPink = pickle.load(pink_hsv_file)
calBlue = pickle.load(blue_hsv_file)
calGreen = pickle.load(green_hsv_file)

stereo_calib_file.close()
kalman_calib_file.close()
world_frame_file.close()
pink_hsv_file.close()
blue_hsv_file.close()
green_hsv_file.close()

calibColors = [calPink, # place in order of points(pt1 -> pink, pt2 -> blue, etc)
               calBlue,
               calGreen]

app, w, SP, LPh, LPr = Plot3Dpoint.HandInitialize()
w, armLines = Plot3Dpoint.ArmInitialize(w)

# begin video capture
c1 = cv2.VideoCapture(0)
c2 = cv2.VideoCapture(1)
cv2.namedWindow('camera 1')
cv2.namedWindow('camera 2')
_,sizing_frame = c1.read()
cv2.moveWindow('camera 1',20,20)
cv2.moveWindow('camera 2',sizing_frame.shape[1]+18+20,20)

# open serial connection to Arduino

connected = False
ser = serial.Serial('COM3', 9600)

while not connected:
    serin = ser.read()
    connected = True

theta = np.array([80, 135, 45, 45, 45, 45, 45])
t_waitmove = 0.5
t_moved = time.time()


##KALMAN INITIALIZE

## point track Kalman

dt = float(1)/30
time1 = time.time()

X = np.zeros((num_points*num_cams,4,2))
A = np.tile(np.eye(4) + dt * np.eye(4,k=2),(num_points*num_cams,1,1))
P = np.tile(100*np.eye(4),(num_points*num_cams,1,1))
Q = 1*np.eye(4)
R = KalmanCalib['R_2D']
H = np.zeros((2,4))
H[:,:2] = np.eye(2)

Z = np.zeros((num_points*num_cams,2,1))

world3D0 = np.zeros((4,num_points))

pos = np.zeros((num_points+1,3))

## 3d triangulation Kalman

X_3D = np.zeros((num_points,6,2))
A_3D = np.tile(np.eye(6) + dt * np.eye(6,k=3),(num_points,1,1))
P_3D = np.tile(100*np.eye(6),(num_points,1,1))
Q_3D = 1*np.eye(6)
R_3D = KalmanCalib['R_3D']
H_3D = np.zeros((3,6))
H_3D[:,:3] = np.eye(3)

Z_3D = np.zeros((num_points,3,1))

## END KALMAN INITIALIZE

frame = [np.empty((sizing_frame.shape[0],sizing_frame.shape[1],3)),
         np.empty((sizing_frame.shape[0],sizing_frame.shape[1],3))]

while 1:
    _,frame[0] = c1.read()
    _,frame[1] = c2.read()

    for i in range(num_cams):
        for j in range(num_points):
            Z[num_points*i + j,:] = np.asarray([pointTrack(frame[i],calibColors[j])]).T

    ## KALMAN
    
    dt = time.time() - time1
    time1 = time.time()

    for i in range(num_cams):
        for j in range(num_points):
            A = np.eye(4) + dt * np.eye(4,k=2)
            Xi = X[num_points*i + j,:,:]
            Pi = P[num_points*i + j,:,:]
            Zi = Z[num_points*i + j,:,:]
            (Xi,Pi) = Kalman.predict(Xi,Pi,A,Q)
            if np.any(Zi):
                (Xi,Pi) = Kalman.update(Pi,H,R,Xi,Zi)
                
            X[num_points*i + j,:,:] = Xi
            P[num_points*i + j,:,:] = Pi
            Z[num_points*i + j,:,:] = Zi

    ## END KALMAN

    Z_disp = np.int32(np.around(Z))
    X_disp = np.int32(np.around(X))

    cv2.circle(frame[0],(Z_disp[0,0,0],Z_disp[0,1,0]),5,255,-1)
    cv2.circle(frame[0],(Z_disp[1,0,0],Z_disp[1,1,0]),5,255,-1)
    cv2.circle(frame[0],(Z_disp[2,0,0],Z_disp[2,1,0]),5,255,-1)

    cv2.circle(frame[1],(Z_disp[3,0,0],Z_disp[3,1,0]),5,255,-1)
    cv2.circle(frame[1],(Z_disp[4,0,0],Z_disp[4,1,0]),5,255,-1)
    cv2.circle(frame[1],(Z_disp[5,0,0],Z_disp[5,1,0]),5,255,-1)

    cv2.circle(frame[0],(X_disp[0,0,0],X_disp[0,1,0]),10,255,2)
    cv2.circle(frame[0],(X_disp[1,0,0],X_disp[1,1,0]),10,255,2)
    cv2.circle(frame[0],(X_disp[2,0,0],X_disp[2,1,0]),10,255,2)

    cv2.circle(frame[1],(X_disp[3,0,0],X_disp[3,1,0]),10,255,2)
    cv2.circle(frame[1],(X_disp[4,0,0],X_disp[4,1,0]),10,255,2)
    cv2.circle(frame[1],(X_disp[5,0,0],X_disp[5,1,0]),10,255,2)

    cv2.imshow('camera 1',frame[0])
    cv2.imshow('camera 2',frame[1])

    projPoints1 = np.asarray([(X[0,0,0],X[0,1,0]),
                              (X[1,0,0],X[1,1,0]),
                              (X[2,0,0],X[2,1,0])])
    projPoints2 = np.asarray([(X[3,0,0],X[3,1,0]),
                              (X[4,0,0],X[4,1,0]),
                              (X[5,0,0],X[5,1,0])])

    points3D = TriangulatePoint.linearTriangulate(StereoCalib['P_R'], StereoCalib['P_L'],
                                                 projPoints1.T, projPoints2.T)

    for i in range(num_points):
        Z_3D[i,:] = np.asarray([np.dot(TransformMat,np.append(points3D[:,i],1))[:3]]).T

    ## 3D Triangulation Kalman

    for i in range(num_points):
        A_3D = np.eye(6) + dt * np.eye(6,k=3)
        Xi_3D = X_3D[i,:,:]
        Pi_3D = P_3D[i,:,:]
        Zi_3D = Z_3D[i,:,:]
        (Xi_3D,Pi_3D) = Kalman.predict(Xi_3D,Pi_3D,A_3D,Q_3D)
        (Xi_3D,Pi_3D) = Kalman.update(Pi_3D,H_3D,R_3D,Xi_3D,Zi_3D)

        X_3D[i,:,:] = Xi_3D
        P_3D[i,:,:] = Pi_3D
        Z_3D[i,:,:] = Zi_3D

##    world3D = np.reshape(Z_3D,(3,3)).T
    world3D = np.reshape(X_3D[:,:3,:1],(3,3)).T

    print world3D

    ## END KALMAN

    centroid = world3D[:,1]

    headingvec = np.cross(world3D[:,0]-world3D[:,1],world3D[:,0]-world3D[:,2])
    if np.linalg.norm(headingvec) != 0:
        headingvec /= np.linalg.norm(headingvec)
    headingvec_scaled = 75*headingvec
    headingvec_scaled += centroid

    rollvec = world3D[:,2]-world3D[:,1]
    if np.linalg.norm(rollvec) != 0:
        rollvec /= np.linalg.norm(rollvec)
    rollvec_scaled = 50*rollvec
    rollvec_scaled += centroid

    pos[:3,:] = world3D.T
    pos[3,:] = centroid

    Plot3Dpoint.HandUpdate(SP,LPh,LPr,pos,np.array([centroid,headingvec_scaled]),
                           np.array([centroid,rollvec_scaled]))

    Xg = np.empty((6))
    Xg[:3] = centroid
    Xg[3:6] = headingvec

    print Xg
    print ''

##    Xg = np.minimum(Xg,np.array([200,200,200,1,1,1]))
##    Xg = np.maximum(Xg,np.array([0,0,0,0,0,0]))

##    Xg[:3] =  np.abs(Xg[:3])
##    r_X = np.linalg.norm(Xg[:2])
##
##    if r_X > 200:
##        Xg[0] = Xg[0] * 200/r_X
##        Xg[1] = Xg[1] * 200/r_X
##    elif r_X < 100:
##        Xg[0] = Xg[0] * 100/r_X
##        Xg[1] = Xg[1] * 100/r_X
##
##    Xg[2] = np.minimum(Xg[2],200)
##    Xg[2] = np.maximum(Xg[2],100)

    print Xg
    print ''

    ## solve IK and move arm

    theta_arm, arm_pos = IKsolve(theta[:6], Xg)
    theta[:6] = theta_arm

    ## plot arm

    Plot3Dpoint.ArmUpdate(armLines, arm_pos)

    if ((time.time() - t_moved) > t_waitmove):
        t_moved = time.time()
        moveArm(ser,theta)
        print 'moved'

    # press ESC to break loop
    if cv2.waitKey(5)==27:
        break

# destroy windows and release camera after ESC
cv2.destroyAllWindows()
c1.release()
c2.release()
Plot3Dpoint.close(w)
ser.close()
