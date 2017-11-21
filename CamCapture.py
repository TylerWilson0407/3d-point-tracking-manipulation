# import modules
import cv2
import numpy as np

# begin video capture
c = cv2.VideoCapture(0)

# define windows
cv2.namedWindow('HSVvalues') # trackbars for HSV values
cv2.resizeWindow('HSVvalues',500,350)
cv2.namedWindow('camera') # unaltered camera feed
cv2.namedWindow('thresh') # HSV threshold image
cv2.namedWindow('blurred') # blurred image
cv2.namedWindow('thresh_blur')

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

cv2.createTrackbar('H min','HSVvalues',10,100,nothing)
cv2.createTrackbar('H max','HSVvalues',34,100,nothing)
cv2.createTrackbar('S min','HSVvalues',116,255,nothing)
cv2.createTrackbar('S max','HSVvalues',118,255,nothing)
cv2.createTrackbar('V min','HSVvalues',142,255,nothing)
cv2.createTrackbar('V max','HSVvalues',255,255,nothing)
cv2.createTrackbar('Blur k','HSVvalues',10,20,nothing)

while(1):
    _,frame = c.read()
    
    # smooth it
    blur_k = cv2.getTrackbarPos('Blur k','HSVvalues')
    frame_blur = cv2.blur(frame,(blur_k,blur_k))

    # convert to hsv and find range of colors
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    HSVvalues = [cv2.getTrackbarPos('H min','HSVvalues'),
                 cv2.getTrackbarPos('H max','HSVvalues'),
                 cv2.getTrackbarPos('S min','HSVvalues'),
                 cv2.getTrackbarPos('S max','HSVvalues'),
                 cv2.getTrackbarPos('V min','HSVvalues'),
                 cv2.getTrackbarPos('V max','HSVvalues')]
    thresh = cv2.inRange(hsv,np.array((HSVvalues[0], HSVvalues[1], HSVvalues[2])),
                         np.array((HSVvalues[3], HSVvalues[4], HSVvalues[5])))
    thresh2 = thresh.copy()

    # convert to hsv and find range of colors
    hsv_blur = cv2.cvtColor(frame_blur,cv2.COLOR_BGR2HSV)
    HSVvalues = [cv2.getTrackbarPos('H min','HSVvalues'),
                 cv2.getTrackbarPos('H max','HSVvalues'),
                 cv2.getTrackbarPos('S min','HSVvalues'),
                 cv2.getTrackbarPos('S max','HSVvalues'),
                 cv2.getTrackbarPos('V min','HSVvalues'),
                 cv2.getTrackbarPos('V max','HSVvalues')]
    thresh_blur = cv2.inRange(hsv_blur,np.array((HSVvalues[0], HSVvalues[1], HSVvalues[2])),
                         np.array((HSVvalues[3], HSVvalues[4], HSVvalues[5])))
    thresh2_blur = thresh_blur.copy()

    # find contours in the threshold image
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # find contours in the threshold image
    contours,hierarchy = cv2.findContours(thresh_blur,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

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
    cv2.imshow('blurred',frame_blur)
    cv2.imshow('thresh',thresh2)
    cv2.imshow('thresh_blur',thresh2_blur)
##    cv2.imshow('HSVvalues')

    # press ESC to break loop
    if cv2.waitKey(5)==27:
        break

# destroy windows after ESC
cv2.destroyAllWindows()
