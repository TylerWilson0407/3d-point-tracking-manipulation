def StereoCalibFrameCap():
    import cv2
    import numpy as np
    import pickle

    c_R = cv2.VideoCapture(0)
    c_L = cv2.VideoCapture(2)
    _,frame = c_R.read()
    cv2.namedWindow('Right Camera')
    cv2.namedWindow('Left Camera')
    cv2.namedWindow('Right Capture')
    cv2.namedWindow('Left Capture')
    cv2.moveWindow('Right Camera',200,200)
    cv2.moveWindow('Left Camera',frame.shape[1]+18+20,200)
    cv2.moveWindow('Right Capture',200,frame.shape[0]+42+200)
    cv2.moveWindow('Left Capture',frame.shape[1]+18+200,frame.shape[0]+42+200)

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

StereoCalibFrameCap()