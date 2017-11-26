from config import config
import cv2
import numpy as np

im_num = 1
leftright = 'L'
imfile = r'test_images/ballcalib_' + str(im_num) + '_' + leftright + '.bmp'

frame = cv2.imread(imfile)


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


def circle_mouse_callback(event, x, y, flags, params):
    """Mouse callback function for selecting a circle on a frame by dragging
    from the center to the edge"""
    if event == cv2.EVENT_LBUTTONDOWN and not params['drag']:
        params['point1'] = np.array([x, y])
        params['drag'] = True
    elif event == cv2.EVENT_MOUSEMOVE and params['drag']:
        params['point2'] = np.array([x, y])
    elif event == cv2.EVENT_LBUTTONUP and params['drag']:
        params['point2'] = np.array([x, y])
        params['drag'] = False
        if not np.array_equal(params['point2'], params['point1']):
            params['selected'] = True
    return


def select_roi(frame):
    """Select a region of interest from a captured frame"""
    wd_frame = 'frame'
    cv2.namedWindow(wd_frame)
    cv2.moveWindow(wd_frame,
                   config['windows']['ORIGIN_X'],
                   config['windows']['ORIGIN_Y'])

    callback_params = {'drag': None,
                       'frame': frame,
                       'roi': None,
                       'roi_origin': None}

    while not np.any(callback_params['roi']):
        cv2.imshow(wd_frame, frame)

        cv2.setMouseCallback('frame', roi_mouse_callback,
                             param=callback_params)

        if cv2.waitKey(5) == 27:
            break

    return callback_params['roi']


def select_circle(img):
    """Select a region of interest from a captured frame"""
    wd_frame = 'Select Calibration Targets'
    cv2.namedWindow(wd_frame)
    cv2.moveWindow(wd_frame,
                   config['windows']['ORIGIN_X'],
                   config['windows']['ORIGIN_Y'])

    center = None
    radius = None

    cb_params = {'drag': None,
                       'point1': np.zeros(2),
                       'point2': np.zeros(2),
                       'selected': None}

    while not cb_params['selected']:
        cv2.imshow(wd_frame, img)

        cv2.setMouseCallback(wd_frame, circle_mouse_callback,
                             param=cb_params)

        if cb_params['drag']:
            circ_img = img.copy()
            radius = np.int(np.linalg.norm(cb_params['point1'] -
                                           cb_params['point2']))
            cv2.circle(circ_img,
                       tuple(cb_params['point1']),
                       radius,
                       255, 2, 8, 0)
            cv2.imshow(wd_frame, circ_img)

        center = cb_params['point1']
        radius = np.int(np.linalg.norm(cb_params['point1'] -
                                       cb_params['point2']))

        if cv2.waitKey(5) == 27:
            break

    circle = [center,
              radius]
    return circle


def scale_image(image, scale):
    """Resize an image by a given scale."""
    x_scale = int(np.around(image.shape[1] * scale))
    y_scale = int(np.around(image.shape[0] * scale))
    scaled_image = cv2.resize(image, (x_scale, y_scale))
    return scaled_image


def confirm_roi(roi):
    """Confirm selected region of interest.

    Return True if confirmed, False otherwise.
    """
    wd_roi = 'Region of Interest'
    cv2.namedWindow(wd_roi)
    cv2.moveWindow(wd_roi,
                   config['windows']['ORIGIN_X'],
                   config['windows']['ORIGIN_Y'])

    while True:
        cv2.imshow(wd_roi, roi)

        keypress = cv2.waitKey(5)

        if keypress == 13:  # return True if Enter pressed
            cv2.destroyWindow(wd_roi)
            return True

        if keypress == 27:  # return False if Escape pressed
            cv2.destroyWindow(wd_roi)
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

    cf = config['find_circle']

    short_side = min([img.shape[0], img.shape[1]])
    long_side = max([img.shape[0], img.shape[1]])
    max_radius = short_side // 2
    img_blur = cv2.blur(img, (cf['BLUR_K'], cf['BLUR_K']))
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT,
                               dp=cf['HOUGH_DP'],
                               minDist=long_side,
                               param1=cf['HOUGH_PARAM1'],
                               param2=cf['HOUGH_PARAM2'],
                               maxRadius=max_radius)
    ### ADD PRINT STATEMENTS HERE TO SAY WHETHER CIRCLE WAS FOUND OR NOT
    if np.any(circles):
        circle = circles[0][0]
    else:
        circle = np.array([img.shape[1] / 2],
                          [img.shape[1] / 2],
                          short_side * 0.75)
    return circle


def empty_callback(*_, **__):
    """Empty callback function for passing to functions that require a
    callback.  Accepts any argument and returns None.
    """
    return None


def adjust_circle(img, initial_circle):
    """Manually adjust a circle on an image"""

    scale = config['roi']['ADJUST_SCALE']
    img = scale_image(img, scale)
    initial_circle = initial_circle * scale

    wd_adjcir = 'Adjust Circle Target'
    keypress = -1
    img_circ = np.copy(img)
    # Set max radius of circle as half of the longest side of image
    max_radius = np.max([img.shape[0], img.shape[1]]) // 2
    cv2.namedWindow(wd_adjcir)
    cv2.resizeWindow(wd_adjcir, 200, 200)
    cv2.createTrackbar('x', wd_adjcir,
                       initial_circle[0], img.shape[1], empty_callback)
    cv2.createTrackbar('y', wd_adjcir,
                       initial_circle[1], img.shape[0], empty_callback)
    cv2.createTrackbar('r', wd_adjcir,
                       initial_circle[2], max_radius, empty_callback)
    while keypress == -1:
        cv2.circle(img_circ,
                   (cv2.getTrackbarPos('x', wd_adjcir),
                    cv2.getTrackbarPos('y', wd_adjcir)),
                   cv2.getTrackbarPos('r', wd_adjcir),
                   255,
                   1)
        cv2.imshow(wd_adjcir, img_circ)
        img_circ = np.copy(img)
        keypress = cv2.waitKey(5)
    return  # RETURN ADJUSTED CIRCLE PARAMETERS HERE


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


# test_roi = get_roi(frame)
# # circle = find_circle(test_roi)
# circle = get_circle_roi(test_roi)

test_circle = select_circle(frame)
print(test_circle)
test_circle_adj = adjust_circle(frame, test_circle)