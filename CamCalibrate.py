def HSVcalib(cam,pct_low,pct_hi,h_ext,s_ext,v_ext,targetName):
    import cv2
    import numpy as np
    import pickle

    global drag, point1, point2, roi, roiSelected

    # begin video capture
    c = cv2.VideoCapture(cam)
    cv2.namedWindow('camera')

    # variable setup
    drag = 0
    windowPosCheck = 0
    roi = 0
    roiSelected = 0
    roiConfirmed = 0

    # mouse callback function
    def mouseCallFunc(event,x,y,flags,param):
        global drag, point1, point2, roi, roiSelected
        if (event == cv2.EVENT_LBUTTONDOWN and not drag):
            point1 = (x,y)
            drag = 1
        elif (event == cv2.EVENT_MOUSEMOVE and drag):
            img1 = frame.copy()
            cv2.rectangle(img1,point1,(x,y),255,1,8,0)
            cv2.imshow('camera',img1)
        elif (event == cv2.EVENT_LBUTTONUP and drag):
            img1 = frame.copy()
            point2 = (x,y)
            if point2 != point1:
                roi = frame[point1[1]:point2[1], point1[0]:point2[0]]
                roiSelected = 1
            drag = 0

    # while loop until ROI is confirmed
    while 1:
        _,frame = c.read() #read and display current frame
        cv2.imshow('camera',frame)
        cv2.moveWindow('camera',20,20)

        # mouse callback to select ROI
        cv2.setMouseCallback('camera',mouseCallFunc,param=None)

        # while loop once ROI is selected
        while roiSelected:

            # initial blur kernel
            blur_k = 10
            roi_blur = cv2.blur(roi,(blur_k,blur_k))

            # convert to HSV, split and histogram for display
            roi_hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            roi_h = cv2.split(roi_hsv)[0]
            roi_s = cv2.split(roi_hsv)[1]
            roi_v = cv2.split(roi_hsv)[2]

            iHeight,iWidth,iDepth = roi_hsv.shape
            h = np.zeros((iHeight,iWidth,iDepth))
            bins = np.arange(256).reshape(256,1)
            color = [ (255,0,0),(0,255,0),(0,0,255)]

            for ch,col in enumerate(color):
                hist_item = cv2.calcHist([roi_hsv],[ch],None,[256],[0,255])
                cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
                hist=np.int32(np.around(hist_item))
                pts = np.column_stack((bins,hist))
                cv2.polylines(h,[pts],False,col)

            h= np.flipud(h)

            cv2.imshow('crop',roi)
            cv2.imshow('roi hue',roi_h)
            cv2.imshow('roi sat',roi_s)
            cv2.imshow('roi val',roi_v)
            cv2.imshow('colorhist',h)

            # position windows on first pass
            if not windowPosCheck:
                windowPosCheck = 1
                cv2.moveWindow('crop',frame.shape[1]+18*1+20,20)
                cv2.moveWindow('roi hue',frame.shape[1]+roi.shape[1]+18*2+20,20)
                cv2.moveWindow('roi sat',frame.shape[1]+18*1+20,roi.shape[0]+62)
                cv2.moveWindow('roi val',frame.shape[1]+roi.shape[1]+18*2+20,roi.shape[0]+62)
                cv2.moveWindow('colorhist',frame.shape[1]+2*roi.shape[1]+18*3+20,20)
                if h.shape[0] > 256 and h.shape[1] > 256:
                    h = h[0:255,0:255]
                    cv2.resizeWindow('colorhist',256,256)

            # press ESC to confirm selected ROI
            if cv2.waitKey(5)==27:
                roiSelected = 0
                windowPosCheck = 0
                cv2.destroyWindow('crop')
                cv2.destroyWindow('roi hue')
                cv2.destroyWindow('roi sat')
                cv2.destroyWindow('roi val')
                cv2.destroyWindow('colorhist')
                HSVvalues1 = [np.percentile(roi_h,pct_low), np.percentile(roi_h,pct_hi),
                             np.percentile(roi_s,pct_low), np.percentile(roi_s,pct_hi),
                             np.percentile(roi_v,pct_low), np.percentile(roi_v,pct_hi)]

                HSVvalues = np.zeros(6)
                HSVvalues[0] = np.amax((0,HSVvalues1[0] - h_ext*(HSVvalues1[1] - HSVvalues1[0])))
                HSVvalues[1] = np.amin((180,HSVvalues1[1] + h_ext*(HSVvalues1[1] - HSVvalues1[0])))
                HSVvalues[2] = np.amax((0,HSVvalues1[2] - s_ext*(HSVvalues1[3] - HSVvalues1[2])))
                HSVvalues[3] = np.amin((255,HSVvalues1[3] + s_ext*(HSVvalues1[3] - HSVvalues1[2])))
                HSVvalues[4] = np.amax((0,HSVvalues1[4] - v_ext*(HSVvalues1[5] - HSVvalues1[4])))
                HSVvalues[5] = np.amin((255,HSVvalues1[5] + v_ext*(HSVvalues1[5] - HSVvalues1[4])))
                HSVvalues = np.int32(np.around(HSVvalues))
                break

        # press ESC to break loop
        if cv2.waitKey(5)==27:
            break


    try:
        HSVvalues
    except NameError:
        print("Region not selected for calibration")
        cv2.destroyAllWindows()
        c.release()
        raise SystemExit

    ####################################### show image with calibrated HSV values for adjusting

    # define windows
    cv2.namedWindow('HSVvalues') # trackbars for HSV values
    cv2.namedWindow('thresh') # HSV threshold image
    cv2.resizeWindow('HSVvalues',500,350)


    # trackbar for HSV ranges
    def nothing(x):
        if cv2.getTrackbarPos('H min','HSVvalues') > cv2.getTrackbarPos('H max','HSVvalues'):
            cv2.setTrackbarPos('H min','HSVvalues',cv2.getTrackbarPos('H max','HSVvalues'))
        elif cv2.getTrackbarPos('H max','HSVvalues') < cv2.getTrackbarPos('H min','HSVvalues'):
            cv2.setTrackbarPos('H max','HSVvalues',cv2.getTrackbarPos('H min','HSVvalues'))
        elif cv2.getTrackbarPos('S min','HSVvalues') > cv2.getTrackbarPos('S max','HSVvalues'):
            cv2.setTrackbarPos('S min','HSVvalues',cv2.getTrackbarPos('S max','HSVvalues'))
        elif cv2.getTrackbarPos('S max','HSVvalues') < cv2.getTrackbarPos('S min','HSVvalues'):
            cv2.setTrackbarPos('S max','HSVvalues',cv2.getTrackbarPos('S min','HSVvalues'))
        elif cv2.getTrackbarPos('V min','HSVvalues') > cv2.getTrackbarPos('V max','HSVvalues'):
            cv2.setTrackbarPos('V min','HSVvalues',cv2.getTrackbarPos('V max','HSVvalues'))
        elif cv2.getTrackbarPos('V max','HSVvalues') < cv2.getTrackbarPos('V min','HSVvalues'):
            cv2.setTrackbarPos('V max','HSVvalues',cv2.getTrackbarPos('V min','HSVvalues'))
        elif cv2.getTrackbarPos('Blur k','HSVvalues') == 0:
            cv2.setTrackbarPos('Blur k','HSVvalues',1)

    cv2.createTrackbar('H min','HSVvalues',HSVvalues[0],179,nothing)
    cv2.createTrackbar('H max','HSVvalues',HSVvalues[1],179,nothing)
    cv2.createTrackbar('S min','HSVvalues',HSVvalues[2],255,nothing)
    cv2.createTrackbar('S max','HSVvalues',HSVvalues[3],255,nothing)
    cv2.createTrackbar('V min','HSVvalues',HSVvalues[4],255,nothing)
    cv2.createTrackbar('V max','HSVvalues',HSVvalues[5],255,nothing)
    cv2.createTrackbar('Blur k','HSVvalues',10,20,nothing)

    moved = 0

    while(1):
        _,frame = c.read()
        
        # smooth it
        blur_k = cv2.getTrackbarPos('Blur k','HSVvalues')
        frame_blur = cv2.blur(frame,(blur_k,blur_k))

        # convert to hsv and find range of colors
        hsv = cv2.cvtColor(frame_blur,cv2.COLOR_BGR2HSV)
        HSVvalues = [cv2.getTrackbarPos('H min','HSVvalues'),
                     cv2.getTrackbarPos('H max','HSVvalues'),
                     cv2.getTrackbarPos('S min','HSVvalues'),
                     cv2.getTrackbarPos('S max','HSVvalues'),
                     cv2.getTrackbarPos('V min','HSVvalues'),
                     cv2.getTrackbarPos('V max','HSVvalues')]
        thresh = cv2.inRange(hsv,np.array((HSVvalues[0], HSVvalues[2], HSVvalues[4])),
                             np.array((HSVvalues[1], HSVvalues[3], HSVvalues[5])))
        thresh2 = thresh.copy()

        # find contours in the threshold image
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # finding contour with maximum area and store it as best_cnt
        max_area = 0
        best_cnt = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = cnt

        # finding centroids of best_cnt and draw a circle there
        M = cv2.moments(best_cnt)
        cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        cv2.circle(frame,(cx,cy),5,255,-1)

        # Show it
        cv2.imshow('camera',frame)
        cv2.imshow('thresh',thresh2)

        if not moved:
            moved = 1
            cv2.moveWindow('camera',20,20)
            cv2.moveWindow('thresh',frame.shape[1]+38,20)
            cv2.moveWindow('HSVvalues',2*frame.shape[1]+56,20)

        # press ESC to break loop
        if cv2.waitKey(5)==27:
            break

    # destroy windows after ESC
    cv2.destroyAllWindows()
    c.release()

    HSVvalues = [(HSVvalues[0],HSVvalues[2],HSVvalues[4]),
            (HSVvalues[1],HSVvalues[3],HSVvalues[5])]

    output = open('Calibration Data\\' + targetName + '.pkl', 'wb')

    pickle.dump(HSVvalues, output)

    output.close()

    return HSVvalues

def Rcalib(t_sample):
    import cv2
    import numpy as np
    import pickle
    import time
    
    from pointTrack import pointTrack

    # begin video capture
    c1 = cv2.VideoCapture(0)
    c2 = cv2.VideoCapture(1)
    cv2.namedWindow('camera 1')
    cv2.namedWindow('camera 2')
    _,frame = c1.read()
    cv2.moveWindow('camera 1',20,20)
    cv2.moveWindow('camera 2',frame.shape[1]+18+20,20)

    stereo_calib_file = open('Calibration Data\stereocalib.pkl', 'rb')
    blue_hsv_file = open('Calibration Data\HSVblue_1.pkl', 'rb')

    StereoCalib = pickle.load(stereo_calib_file)
    calBlue = pickle.load(blue_hsv_file)

    stereo_calib_file.close()
    blue_hsv_file.close()

    t0 = time.time()
    X = np.empty((2,1))
    i = 0

    while time.time() < (t0 + t_sample):
        _,frame1 = c1.read()
        _,frame2 = c2.read()

##        cv2.imshow('camera 1',frame1)
##        cv2.imshow('camera 2',frame2)

        pt1 = pointTrack(frame1,calBlue)
        pt2 = pointTrack(frame2,calBlue)

        cv2.circle(frame1,pt1,5,255,-1)
        cv2.circle(frame2,pt2,5,255,-1)

        X[:,i] = [pt1[0],pt1[1],pt2[0],pt2[1]]

        i += 1

        # press ESC to break loop
        if cv2.waitKey(5)==27:
            break

    return np.cov(X)
    
def StereoCalibFrameCap():
    import cv2
    import numpy as np
    import pickle

    c_R = cv2.VideoCapture(0)
    c_L = cv2.VideoCapture(1)
    _,frame = c_R.read()
    cv2.namedWindow('Right Camera')
    cv2.namedWindow('Left Camera')
    cv2.namedWindow('Right Capture')
    cv2.namedWindow('Left Capture')
    cv2.moveWindow('Right Camera',20,20)
    cv2.moveWindow('Left Camera',frame.shape[1]+18+20,20)
    cv2.moveWindow('Right Capture',20,frame.shape[0]+42+20)
    cv2.moveWindow('Left Capture',frame.shape[1]+18+20,frame.shape[0]+42+20)

    numFrames = 20
    num_horiz = 8
    num_vert = 6
    square_length = 23.3 # distance between corners in mm
    chess_dims = (num_horiz,num_vert)

    i = 1

    objectPoints = []
    imagePointsR = []
    imagePointsL = []

    while i < numFrames:
        _,frameR = c_R.read()
        _,frameL = c_L.read()

        cv2.imshow('Right Camera',frameR)
        cv2.imshow('Left Camera',frameL)

        # SPACE to capture frame and find chessboard points
        if cv2.waitKey(5)==32:
            _,capR = c_R.read()
            _,capL = c_L.read()

            capR = cv2.cvtColor(capR, cv2.COLOR_BGR2GRAY)
            capL = cv2.cvtColor(capL, cv2.COLOR_BGR2GRAY)

            retR, cornersR = cv2.findChessboardCorners(capR,chess_dims)
            retL, cornersL = cv2.findChessboardCorners(capL,chess_dims)

            cv2.drawChessboardCorners(capR,chess_dims,cornersR,0)
            cv2.drawChessboardCorners(capL,chess_dims,cornersL,0)

            if retR and retL:

                objPoints = np.zeros((num_horiz*num_vert,3), np.float32)
                objPoints[:,:2] = square_length*np.mgrid[0:num_horiz,0:num_vert].T.reshape(-1,2)

                cv2.imwrite('test_images\calib_' + str(i) + '_R.jpg',capR,(cv2.IMWRITE_PNG_COMPRESSION,0))
                cv2.imwrite('test_images\calib_' + str(i) + '_L.jpg',capL,(cv2.IMWRITE_PNG_COMPRESSION,0))

                objectPoints.append(objPoints)
                imagePointsR.append(cornersR)
                imagePointsL.append(cornersL)

                cv2.imshow('Right Capture',capR)
                cv2.imshow('Left Capture',capL)

                i += 1

        if cv2.waitKey(5)==27:
            break

    # destroy windows and release camera after ESC
    cv2.destroyAllWindows()
    c_R.release()
    c_L.release()

    if i==numFrames:

        StereoCalibFrameData = { 'ObjPts': objectPoints,
                                 'ImgPtsR': imagePointsR,
                                 'ImgPtsL': imagePointsL}

        output = open('test_images\stereoframedata.pkl', 'wb')

        pickle.dump(StereoCalibFrameData, output)

        output.close()

def StereoCalib():
    import cv2
    import numpy as np
    import pickle

    # define matrices
    R = [0]
    T = [0]
    E=[]
    F=[]
    Q=np.zeros((4,4))
    R_R = np.zeros((3,3))
    R_L = np.zeros((3,3))
    P_R=np.zeros((3,4))
    P_L =np.zeros((3,4))

    newcameraMatrixR = np.zeros((3,3))
    newcameraMatrixL = np.zeros((3,3))

    # import calibration frame info
    calib_frame_file = open('test_images\stereoframedata.pkl', 'rb')
    StereoFrameData = pickle.load(calib_frame_file)
    calib_frame_file.close()

    objectPoints = StereoFrameData['ObjPts']
    imagePointsR = StereoFrameData['ImgPtsR']
    imagePointsL = StereoFrameData['ImgPtsL']

    # capture frame for frame shape
    c_R = cv2.VideoCapture(0)
    c_L = cv2.VideoCapture(1)

    _,frameR = c_R.read()
    _,frameL = c_L.read()

    cv2.namedWindow('Stereo Images')
    cv2.moveWindow('Stereo Images',20,20)

    retval, cameraMatrixR, distCoeffsR, rvecs, tvecs = \
            cv2.calibrateCamera(objectPoints, imagePointsR, (frameR.shape[1],frameR.shape[0]))

    retval, cameraMatrixL, distCoeffsL, rvecs, tvecs = \
            cv2.calibrateCamera(objectPoints, imagePointsL, (frameR.shape[1],frameR.shape[0]))

    retval,cameraMatrixR, distCoeffsR, cameraMatrixL, distCoeffsL, R, T, E, F = \
                              cv2.stereoCalibrate(objectPoints, imagePointsR, imagePointsL,
                                                  (frameR.shape[1],frameR.shape[0]),
                                                  cameraMatrixR, distCoeffsR,
                                                  cameraMatrixL, distCoeffsL,
                                                  flags=cv2.CALIB_FIX_INTRINSIC)

    cv2.stereoRectify(cameraMatrixR, distCoeffsR, cameraMatrixL, distCoeffsL,
                      (frameR.shape[1],frameR.shape[0]), R, T, R_R, R_L, P_R, P_L,
                      Q, 0, -1, (0, 0))

    mapR_1,mapR_2 = cv2.initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R_R,
                                              P_R, (frameR.shape[1],frameR.shape[0]),
                                              cv2.CV_32FC1)

    mapL_1,mapL_2 = cv2.initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R_L,
                                              P_L, (frameL.shape[1],frameL.shape[0]),
                                              cv2.CV_32FC1)

    while 1:

        _,capR = c_R.read()
        _,capL = c_L.read()

        dstR = cv2.remap(capR, mapR_1, mapR_2, cv2.INTER_LINEAR)
        dstL = cv2.remap(capL, mapL_1, mapL_2, cv2.INTER_LINEAR)

        dstLR = np.zeros((max(dstR.shape[0],dstL.shape[0]),dstR.shape[1]+dstL.shape[1],3),np.uint8)
        dstLR[:dstR.shape[0],:dstR.shape[1]] = dstR
        dstLR[:dstL.shape[0],dstR.shape[1]:dstR.shape[1]+dstL.shape[1]] = dstL

        for i in range(0,dstLR.shape[0],15):
            cv2.line(dstLR, (0,i), (dstLR.shape[1],i), (0,0,255), 1, 8, 0)
        
        cv2.imshow('Stereo Images',dstLR)
        
        # press ESC to break loop
        if cv2.waitKey(5)==27:
            break

    cv2.destroyAllWindows()
    c_R.release()
    c_L.release()

    # save stereo camera calibration info

    StereoCalibrationData = {'CamMat1': cameraMatrixR,
                             'CamMat2': cameraMatrixL,
                             'DistCoeff1': distCoeffsR,
                             'DistCoeff2': distCoeffsL,
                             'R': R,
                             'T': T,
                             'E': E,
                             'F': F,
                             'R_R': R_R,
                             'R_L': R_L,
                             'P_R': P_R,
                             'P_L': P_L,
                             'mapR_1': mapR_1,
                             'mapR_2': mapR_2,
                             'mapL_1': mapL_1,
                             'mapL_2': mapL_2}

    output = open('Calibration Data\stereocalib.pkl', 'wb')

    pickle.dump(StereoCalibrationData, output)

    output.close()

def WorldFrameCalib():
    import cv2
    import numpy as np
    import pickle
    
    import TriangulatePoint

    stereo_calib_file = open('Calibration Data\stereocalib.pkl', 'rb')
    StereoCalib = pickle.load(stereo_calib_file)

    c_R = cv2.VideoCapture(0)
    c_L = cv2.VideoCapture(1)
    _,frame = c_R.read()
    cv2.namedWindow('Right Camera')
    cv2.namedWindow('Left Camera')
    cv2.namedWindow('Right Capture')
    cv2.namedWindow('Left Capture')
    cv2.moveWindow('Right Camera',20,20)
    cv2.moveWindow('Left Camera',frame.shape[1]+18+20,20)
    cv2.moveWindow('Right Capture',20,frame.shape[0]+42+20)
    cv2.moveWindow('Left Capture',frame.shape[1]+18+20,frame.shape[0]+42+20)

    num_horiz = 8
    num_vert = 6
    square_length = 23.18 # distance between corners in mm
    chess_dims = (num_horiz,num_vert)

    cornersR_reshape = np.empty((num_horiz*num_vert,2))
    cornersL_reshape = np.empty((num_horiz*num_vert,2))

    while 1:
        _,frameR = c_R.read()
        _,frameL = c_L.read()

        cv2.imshow('Right Camera',frameR)
        cv2.imshow('Left Camera',frameL)

        if cv2.waitKey(5)==32:
            _,capR = c_R.read()
            _,capL = c_L.read()

            capR = cv2.cvtColor(capR, cv2.COLOR_BGR2GRAY)
            capL = cv2.cvtColor(capL, cv2.COLOR_BGR2GRAY)

            retR, cornersR = cv2.findChessboardCorners(capR,chess_dims)
            retL, cornersL = cv2.findChessboardCorners(capL,chess_dims)

            cv2.drawChessboardCorners(capR,chess_dims,cornersR,0)
            cv2.drawChessboardCorners(capL,chess_dims,cornersL,0)

            for i in range(num_horiz*num_vert):
                cornersR_reshape[i] = (cornersR[i][0][0],cornersR[i][0][1])
                cornersL_reshape[i] = (cornersL[i][0][0],cornersL[i][0][1])

            indexO = num_horiz*num_vert - 1
            indexX = num_horiz*(num_vert-1)
            indexY = num_horiz-1
            
            cv2.circle(capR,tuple(np.int32(np.around(cornersR_reshape[indexO]))),10,255,2)
            cv2.circle(capL,tuple(np.int32(np.around(cornersL_reshape[indexO]))),10,255,2)

            cv2.circle(capR,tuple(np.int32(np.around(cornersR_reshape[indexX]))),10,255,2)
            cv2.circle(capL,tuple(np.int32(np.around(cornersL_reshape[indexX]))),10,255,2)

            cv2.circle(capR,tuple(np.int32(np.around(cornersR_reshape[indexY]))),10,255,2)
            cv2.circle(capL,tuple(np.int32(np.around(cornersL_reshape[indexY]))),10,255,2)

            cv2.imshow('Right Capture',capR)
            cv2.imshow('Left Capture',capL)

            axisptsR = np.asarray([cornersR_reshape[indexO],
                                   cornersR_reshape[indexX],
                                   cornersR_reshape[indexY]])
            axisptsL = np.asarray([cornersL_reshape[indexO],
                                   cornersL_reshape[indexX],
                                   cornersL_reshape[indexY]])

            axispts3D = TriangulatePoint.linearTriangulate(StereoCalib['P_R'], StereoCalib['P_L'],
                                                           axisptsR.T, axisptsL.T)
            axispts3D = axispts3D.T

            print axispts3D

            Xaxis = axispts3D[1,:] - axispts3D[0,:]
            Xaxis = -Xaxis / np.linalg.norm(Xaxis)

            Yaxis = axispts3D[2,:] - axispts3D[0,:]
            Yaxis = -Yaxis / np.linalg.norm(Yaxis)

            Zaxis = np.cross(Xaxis,Yaxis)

            Yaxis = np.cross(Zaxis,Xaxis)

            print Xaxis
            print Yaxis
            print Zaxis

            RotMatrix = np.eye(4)
            RotMatrix[0,:3] = Xaxis
            RotMatrix[1,:3] = Yaxis
            RotMatrix[2,:3] = Zaxis

            print RotMatrix

            TranslateMatrix = np.eye(4)
            TranslateMatrix[:3,3] = -axispts3D[0,:].T

            print TranslateMatrix

            TransformMat = np.dot(RotMatrix,TranslateMatrix)

            print TransformMat

        if cv2.waitKey(5)==27:
            break

    cv2.destroyAllWindows()
    c_R.release()
    c_L.release()

    output = open('Calibration Data\worldtransformmatrix.pkl', 'wb')

    pickle.dump(TransformMat, output)

    output.close()

StereoCalibFrameCap()