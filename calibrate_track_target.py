from config import config
import cv2
import numpy as np


class TrackingTarget:
    def __init__(self, camera):
        self.c = cv2.VideoCapture(camera)
        self._cal_count = None
        self._cal_funcs = None
        self._cal_args = None
        self._cal_init = None
        self._cal_image = None

        self.cb_params = {'drag': None,
                          'point1': np.zeros(2),
                          'point2': np.zeros(2),
                          'released': None}

        self._roi = None
        self._roi_origin = None

        self._rough_circle = None

        self.thresh_percs = config['thresh_percs']

        self.circle = None
        self.thresholds = None

    def calibrate(self):
        """Calibrate the tracking target.  Captures an image from the video
        feed and allows the user to drag a circular region of interest where
        the target is and finely adjust it.  Once the circular ROI is
        selected, an upper and lower percentile of the HSV values of the
        region of interested are calculated, which are used in threshold
        masks to find the target and subsequently track it."""
        self._cal_count = 0

        """List of functions.  ***explain why using a list with cal_count to be
        able to jump back and forth in the calibration steps, as well as the 
        cal_init list"""
        self._cal_funcs = [self.capture_image,
                           self.drag_circle,
                           self.adjust_circle,
                           self.adjust_hsv_thresholds]

        while 0 <= self._cal_count < len(self._cal_funcs):
            self._cal_funcs[self._cal_count]()

    def capture_image(self):
        """Displays video from input camera feed and capture calibration
        frame"""

        instruct = 'Hold up calibration target(s) and press SPACE to ' \
                   'capture image.'
        self.print_instruct(instruct)

        # initialize window
        win = 'Camera Feed'
        cv2.namedWindow(win)
        cv2.moveWindow(win,
                       config['windows']['ORIGIN_X'],
                       config['windows']['ORIGIN_Y'])

        # initialize variables
        keypress = -1

        while keypress == -1:
            __, self._cal_image = self.c.read()
            cv2.imshow(win, self._cal_image)
            keypress = cv2.waitKey(10)

        # adjust procedure counter
        self.keypress_go_to(keypress)

        cv2.destroyWindow(win)
        return

    def drag_circle(self):
        """Select a region of interest from a captured frame"""

        instruct = 'Drag a circle from the center of desired target to the ' \
                   'edge and release mouse.  Circle can be finely adjusted ' \
                   'in next step, so it does not have to be perfect.'
        self.print_instruct(instruct)

        # initialize window
        win = 'Select Calibration Targets'
        cv2.namedWindow(win)
        cv2.moveWindow(win,
                       config['windows']['ORIGIN_X'],
                       config['windows']['ORIGIN_Y'])

        # initialize variables
        self.cb_params['released'] = False
        center = None
        radius = None
        keypress = -1

        cv2.imshow(win, self._cal_image)

        while keypress == -1:

            # mouse callback function for dragging circle on window
            cv2.setMouseCallback(win, circle_mouse_callback,
                                 param=self)

            center = self.cb_params['point1']
            radius = np.int(np.linalg.norm(self.cb_params['point1'] -
                                           self.cb_params['point2']))

            # continuously draw circle on image while mouse is being dragged
            if self.cb_params['drag'] or self.cb_params['released']:
                circ_img = self._cal_image.copy()
                cv2.circle(circ_img,
                           tuple(center),
                           radius,
                           (0, 0, 0), 1, 8, 0)
                cv2.imshow(win, circ_img)

            keypress = cv2.waitKey(5)

        self._rough_circle = [center, radius]

        # adjust procedure counter
        self.keypress_go_to(keypress)

        cv2.destroyWindow(win)
        return

    def adjust_circle(self):
        """Manually adjust a circle on an image"""

        instruct = 'Adjust the circle so that it is co-radial with the ' \
                   'tracking target.'
        self.print_instruct(instruct)

        # initialize window and trackbars
        win = 'Adjust Target Circle'
        cv2.namedWindow(win)
        cv2.resizeWindow(win, 200, 200)

        # initialize variables
        roi, roi_origin = get_circle_roi(self._cal_image, self._rough_circle)

        circle_local = np.copy(self._rough_circle)
        circle_local[0] = self._rough_circle[0] - np.flipud(roi_origin)

        # scale image to be bigger and allow for easier adjustment
        scale = config['roi']['ADJUST_SCALE']
        roi = scale_image(roi, scale)
        circle_local = np.multiply(circle_local, scale)

        img_circ = np.copy(roi)
        # Set max radius of circle such that the max diameter is the length
        # of the region of interest
        max_radius = roi.shape[0] // 2

        # initialize trackbars
        cv2.createTrackbar('x', win,
                           circle_local[0][0], roi.shape[1], empty_callback)
        cv2.createTrackbar('y', win,
                           circle_local[0][1], roi.shape[0], empty_callback)
        cv2.createTrackbar('r', win,
                           circle_local[1], max_radius, empty_callback)

        keypress = -1

        while keypress == -1:
            cv2.circle(img_circ,
                       (cv2.getTrackbarPos('x', win),
                        cv2.getTrackbarPos('y', win)),
                       cv2.getTrackbarPos('r', win),
                       (0, 0, 0),
                       1)
            cv2.imshow(win, img_circ)
            img_circ = np.copy(roi)
            keypress = cv2.waitKey(5)

        self.circle = ((cv2.getTrackbarPos('x', win) // scale +
                        roi_origin[1],
                        cv2.getTrackbarPos('y', win) // scale +
                        roi_origin[0]),
                       cv2.getTrackbarPos('r', win) // scale)

        # adjust procedure counter
        self.keypress_go_to(keypress)

        cv2.destroyWindow(win)
        return

    def adjust_hsv_thresholds(self):
        """Displays thresholded HSV binary and allows user to adjust
        threshold limits."""

        instruct = 'Adjust the HSV threshold limits until the target is ' \
                   'highly visible and the rest of the image is mostly masked.'
        kp_reset = 'Press \'r\' to reset trackbar positions.'
        self.print_instruct(instruct, kp_before=kp_reset)

        # initialize window
        win_track = 'Adjustment Control Panel'
        cv2.namedWindow(win_track)

        # initialize variables and trackbars
        masked_circle = circle_mask(self._cal_image, self.circle)
        self.thresholds = get_hsv_thresholds(masked_circle, self.thresh_percs)

        thresh_names = ['H_LO', 'H_HI',
                        'S_LO', 'S_HI',
                        'V_LO', 'V_HI']

        blur_k_name = 'BLUR \'K\''

        max_vals = [179, 179, 255, 255, 255, 255]

        for thresh, val, max_val in zip(thresh_names,
                                        self.thresholds,
                                        max_vals):
            cv2.createTrackbar(thresh, win_track, val, max_val, empty_callback)

        cv2.createTrackbar(blur_k_name, win_track,
                           config['blur_k']['initial'],
                           config['blur_k']['max'],
                           empty_callback)
        cv2.setTrackbarMin(blur_k_name, win_track, 1)

        keypress = -1

        while keypress == -1:
            __, frame = self.c.read()

            # blur frame
            blur_k = cv2.getTrackbarPos(blur_k_name, win_track)
            # set blur_k to 1 if less than 1, since trackbar lower limit is 0.
            if blur_k < 1:
                blur_k = 1
            frame_blur = cv2.blur(frame, (blur_k, blur_k))

            frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

            thresh_vals = np.asarray([cv2.getTrackbarPos(name, win_track)
                                      for name in thresh_names])

            frame_thresh = cv2.inRange(frame_hsv,
                                       thresh_vals[::2],
                                       thresh_vals[1::2])

            # cv2.imshow(win_thresh, frame_thresh)
            cv2.imshow(win_track, frame_thresh)

            keypress = cv2.waitKey(5)

        # adjust procedure counter
        self.keypress_go_to(keypress)

        cv2.destroyWindow(win_track)

    def keypress_go_to(self, keypress):
        """Print a message and adjust procedure counter based on keyboard
        input(SPACE, ESC, or 'Q').  Option parameters allow for custom
        messages to print, otherwise a default is used

        SPACE increments the counter to proceed to the next step.
        ESC decrements the counter to proceed to the previous step.
        'q' sets the counter to -1 which will exit the loop to abort the
        procedure.

        """
        if keypress == 32:  # SPACE
            if self._cal_count == (len(self._cal_funcs) - 1):
                print('SPACE pressed.  Calibration procedure finished.')
            else:
                print('SPACE pressed.  Continuing to next step.')
            self._cal_count += 1
        elif keypress == 27:  # ESC
            if self._cal_count == 0:
                print('ESC pressed.  Calibration aborted.')
            else:
                print('ESC pressed.  Returning to previous step.')
            self._cal_count -= 1
        elif keypress == 113:  # 'q'
            print('\'q\' pressed.  Calibration aborted.')
            self._cal_count = -1

    def print_instruct(self, message, kp_before=None, kp_after=None):
        """Prints instructions for calibration procedure step."""
        print('*' * 79)
        print(message)
        print('')

        if kp_before:
            print(kp_before)

        if self._cal_count == len(self._cal_funcs):
            spc = 'complete calibration.'
        else:
            spc = 'proceed to next step.'
        print('Press SPACE to accept and ' + spc)

        if self._cal_count == 0:
            esc = 'or \'q\' to abort and end calibration procedure.'
        else:
            esc = 'to return to previous step.\n' \
                  'Press \'q\' to abort calibration procedure.'
        print('Press ESC ' + esc)

        if kp_after:
            print(kp_after)

        print('*' * 79)

        return


# noinspection PyUnusedLocal
def circle_mouse_callback(event, x, y, flags, params):
    """Mouse callback function for selecting a circle on a frame by dragging
    from the center to the edge"""
    if event == cv2.EVENT_LBUTTONDOWN and not params.cb_params['drag']:
        params.cb_params['point1'] = np.array([x, y])
        params.cb_params['drag'] = True
    elif event == cv2.EVENT_MOUSEMOVE and params.cb_params['drag']:
        params.cb_params['point2'] = np.array([x, y])
    elif event == cv2.EVENT_LBUTTONUP and params.cb_params['drag']:
        params.cb_params['point2'] = np.array([x, y])
        params.cb_params['drag'] = False
        params.cb_params['released'] = True
    return


def scale_image(image, scale):
    """Resize an image by a given scale."""
    x_scale = int(np.around(image.shape[1] * scale))
    y_scale = int(np.around(image.shape[0] * scale))
    scaled_image = cv2.resize(image, (x_scale, y_scale))
    return scaled_image


def empty_callback(*_, **__):
    """Empty callback function for passing to functions that require a
    callback.  Accepts any argument and returns None.
    """
    return None


def adjust_circle(roi, roi_origin, initial_circle):
    """Manually adjust a circle on an image"""

    circle_local = initial_circle
    circle_local[0][0] = initial_circle[0][0] - roi_origin[1]
    circle_local[0][1] = initial_circle[0][1] - roi_origin[0]

    scale = config['roi']['ADJUST_SCALE']
    roi = scale_image(roi, scale)
    circle_local = np.multiply(circle_local, scale)

    wd_adjcir = 'Adjust Target Circle'
    keypress = -1
    img_circ = np.copy(roi)
    # Set max radius of circle as half of the longest side of image
    max_radius = np.max([roi.shape[0], roi.shape[1]]) // 2
    cv2.namedWindow(wd_adjcir)
    cv2.resizeWindow(wd_adjcir, 200, 200)
    cv2.createTrackbar('x', wd_adjcir,
                       circle_local[0][0], roi.shape[1], empty_callback)
    cv2.createTrackbar('y', wd_adjcir,
                       circle_local[0][1], roi.shape[0], empty_callback)
    cv2.createTrackbar('r', wd_adjcir,
                       circle_local[1], max_radius, empty_callback)
    while keypress == -1:
        cv2.circle(img_circ,
                   (cv2.getTrackbarPos('x', wd_adjcir),
                    cv2.getTrackbarPos('y', wd_adjcir)),
                   cv2.getTrackbarPos('r', wd_adjcir),
                   (0, 0, 0),
                   1)
        cv2.imshow(wd_adjcir, img_circ)
        img_circ = np.copy(roi)
        keypress = cv2.waitKey(5)

    if keypress == 27:
        return None

    adj_circle = ((cv2.getTrackbarPos('x', wd_adjcir) // scale +
                   roi_origin[1],
                   cv2.getTrackbarPos('y', wd_adjcir) // scale +
                   roi_origin[0]),
                  cv2.getTrackbarPos('r', wd_adjcir) // scale)

    cv2.destroyWindow(wd_adjcir)
    return adj_circle


def get_circle_roi(image, circle):
    """Get a region of interest around a defined circle in an image."""
    center = circle[0]
    radius = circle[1]
    padding = config['roi']['PADDING']
    rad_padded = int(radius * padding)

    y_min = max((center[1] - rad_padded), 0)
    y_max = min((center[1] + rad_padded), image.shape[0])
    x_min = max((center[0] - rad_padded), 0)
    x_max = min((center[0] + rad_padded), image.shape[1])

    roi = image[y_min: y_max, x_min: x_max]

    roi_origin = np.array([y_min, x_min])

    return roi, roi_origin


def get_hsv_thresholds(image, percentiles):
    """Find an upper and lower threshold for each HSV channel given an
    image and upper and lower percentiles."""

    if len(percentiles) == 1:
        percentiles = [percentiles] * image.shape[2]
    elif len(percentiles) != image.shape[2]:
        raise IndexError('Number of percentile ranges must be either 1 or '
                         'match number of channels in image.')

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_thresholds = []
    for channel, perc in zip(cv2.split(image_hsv), percentiles):
        hsv_thresholds.append(channel_percentile(channel, perc))

    # flatten list
    flat_thresh = [item for sublist in hsv_thresholds for item in sublist]

    return flat_thresh


def circle_mask(image, circle):
    """Create a binary mask where all points outside the input circle are
    set to zero(black).
    """
    center = circle[0]
    radius = circle[1]

    padding = config['roi']['PADDING']
    pad_radius = int(padding * radius)

    mask = np.zeros_like(image)
    cv2.circle(mask, center, radius, (1, 1, 1), -1)
    # reversed because circle center is indexed (x, y) where image is (y, x)
    roi_min = reversed(np.subtract(center, pad_radius))
    roi_max = reversed(np.add(center, pad_radius))
    frame_min = [0, 0]
    frame_max = image.shape
    mins = [max(*v) for v in zip(roi_min, frame_min)]
    maxs = [min(*v) for v in zip(roi_max, frame_max)]
    masked = image * mask
    return masked[mins[0]: maxs[0], mins[1]: maxs[1]]


def channel_percentile(channel, percentile, rem_zero=True):
    """Returns a percentile value for a single-channel image.  Optional
    parameter rem_zero will remove all zero values before calculating the
    percentile."""
    flat_vals = np.ndarray.flatten(channel)
    sorted_vals = np.sort(flat_vals)

    if rem_zero:
        values = np.trim_zeros(sorted_vals)
    else:
        values = sorted_vals

    perc_vals = []
    for perc in percentile:
        perc_vals.append(np.int(np.percentile(values, perc)))

    return perc_vals


target = TrackingTarget(0)
target.calibrate()
