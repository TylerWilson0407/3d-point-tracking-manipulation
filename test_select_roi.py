import cv2
import numpy as np
import pickle

im_num = 1
leftright = 'L'
imfile = r'test_images\ballcalib_' + str(im_num) + '_' + leftright + '.bmp'

frame = cv2.imread(imfile)


# noinspection PyUnusedLocal
def roi_mouse_callback(event, x, y, flags, params):
    """Mouse callback function for selecting a region on interest on a frame"""
    if event == cv2.EVENT_LBUTTONDOWN and not params['drag']:
        params['roi_origin'] = (x, y)
        params['drag'] = True
    elif event == cv2.EVENT_MOUSEMOVE and params['drag']:
        rect_frame = params['frame'].copy()
        cv2.rectangle(rect_frame, params['roi_origin'], (x, y), 255, 2, 8, 0)
        cv2.imshow('frame', rect_frame)
    elif event == cv2.EVENT_LBUTTONUP and params['drag']:
        point2 = (x, y)
        params['drag'] = False
        if point2 != params['roi_origin']:
            params['roi'] = params['frame'][params['roi_origin'][1]:point2[1],
                                            params['roi_origin'][0]:point2[0]]
    return


def select_roi(frame):
    """Select a region of interest from a captured frame"""
    cv2.namedWindow('frame')
    cv2.moveWindow('frame', 20, 20)

    callback_params = {'drag': None,
                       'frame': frame,
                       'roi': None,
                       'roi_origin': None}

    while not np.any(callback_params['roi']):
        cv2.imshow('frame', frame)

        cv2.setMouseCallback('frame', roi_mouse_callback,
                             param=callback_params)

        if cv2.waitKey(5) == 27:
            break

    return callback_params['roi']


def confirm_roi(roi):
    """Confirm selected region of interest.

    Return True if confirmed, False otherwise.
    """
    cv2.namedWindow('roi')
    cv2.moveWindow('roi', 20, 20)
    while True:
        cv2.imshow('roi', roi)

        keypress = cv2.waitKey(5)

        if keypress == 13:  # return True if Enter pressed
            cv2.destroyWindow('roi')
            return True

        if keypress == 27:  # return False if Escape pressed
            cv2.destroyWindow('roi')
            return False


def get_roi(frame):
    """Get selected region of interest from input frame"""
    confirmed = False
    roi = None
    while not confirmed:
        roi = select_roi(frame)
        if np.any(roi):
            confirmed = confirm_roi(roi)
        if cv2.waitKey(250) == 27:
            break
    cv2.destroyAllWindows()
    return roi

def find_circle(img):
    """Find largest circle in input image."""
    max_radius = np.int8(np.around(np.min([img.shape[0], img.shape[1]]))
                             / 2)
    long_side = np.min([img.shape[0], img.shape[1]])
    # REPLACE (5,5) WITH SETTINGS VALUES
    img_blur = cv2.blur(img, (5, 5))
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # REPLACE HARDCODES WITH SETTINGS VALUES
    circle = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=long_side + 1,
                               param1=150, param2=1,
                               minRadius=max_radius // 2,
                               maxRadius=max_radius)
    return circle[0][0]

class RoiCircle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def x_incr(self, incr):
        self.x += incr

    def y_incr(self, incr):
        self.y += incr

    def r_incr(self, incr):
        self.r += incr

def adjust_circle(img, initial_circle):
    """Manually adjust a circle on an image"""
    keypress = -1
    img_circ = np.copy(img)
    max_radius = np.max([img.shape[0], img.shape[1]]) // 2
    cv2.namedWindow('Adjust circle')
    # empty callback function
    callback = lambda *_, **__: None
    cv2.createTrackbar('x', 'Adjust circle',
                       initial_circle[0], img.shape[1], callback)
    cv2.createTrackbar('y', 'Adjust circle',
                       initial_circle[1], img.shape[0], callback)
    cv2.createTrackbar('r', 'Adjust circle',
                       initial_circle[2], max_radius, callback)
    while keypress == -1:
        cv2.circle(img_circ,
                   (cv2.getTrackbarPos('x', 'Adjust circle'),
                   cv2.getTrackbarPos('y', 'Adjust circle')),
                   cv2.getTrackbarPos('r', 'Adjust circle'),
                   255,
                   1)
        cv2.imshow('Adjust circle', img_circ)
        img_circ = np.copy(img)
        keypress = cv2.waitKey(5)
    return # RETURN ADJUSTED CIRCLE PARAMETERS HERE


def get_circle_roi(roi):
    """Get circle from input region of interest.  First attempts to find
    circle using Hough Transform, then displays it and allows manual
    adjustment
    """
    init_circ = find_circle(roi)
    circle = adjust_circle(roi, init_circ)
    return circle

def circle_mask(img, center, radius):
    """Creates a circular mask that masks all points outside the defined
    circle.
    """
    pass

test_roi = get_roi(frame)
# circle = find_circle(test_roi)
circle = get_circle_roi(test_roi)