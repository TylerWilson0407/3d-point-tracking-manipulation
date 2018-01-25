from config import config
import cv2
import imutils
import numpy as np
from subprocess import call
import sys
import time


class Target:
    """DOCSTRING"""

    def __init__(self, cam_id):
        """DOCSTRING"""
        self._cam = cam_id
        self._vid_feed = None

        # parameters for calibration function
        self._cal_count = None
        self._cal_funcs = [self._get_cal_frame,
                           self._get_rough_circle,
                           self._get_final_circle,
                           self._get_thresholds]

        # parameters used to pass data between calibration methods
        self._cal_image = None
        self._rough_circle = None
        self.circle = None

        self.thresholds = None

        # attribute for checking if target has been calibrated
        self.calibrated = None

    def calibrate(self, image=np.empty(0)):
        """Calibrate the tracking target.  Captures an image from the video
        feed and allows the user to drag a circular region of interest where
        the target is and finely adjust it.  Once the circular ROI is
        selected, an upper and lower percentile of the HSV values of the
        region of interested are calculated, which are used in threshold
        masks to find the target and subsequently track it.

        If an image is passed as a parameter, method skips the image capture
        step and instead uses the input image as the calibration image.
        """

        self.calibrated = False
        self._vid_feed = cv2.VideoCapture(self._cam)

        if not image.size:
            start_count = self._cal_count = 0
        else:
            start_count = self._cal_count = 1
            self._cal_image = image

        while start_count <= self._cal_count < len(self._cal_funcs):
            self._cal_funcs[self._cal_count]()

        self._vid_feed.release()
        self.calibrated = True

        return

    def _get_cal_frame(self):
        """Displays video from input camera feed and capture calibration
        frame"""

        instruct = 'Hold up calibration target(s) and press SPACE to ' \
                   'capture image.'
        self._print_instruct(instruct)

        self._cal_image, keypress = capture_image(self._vid_feed)

        # adjust procedure counter
        self._keypress_go_to(keypress)

        return

    def _get_rough_circle(self):
        """Select a region of interest from a captured frame"""

        instruct = 'Drag a circle from the center of desired target to the ' \
                   'edge and release mouse.  Circle can be finely adjusted ' \
                   'in next step, so it does not have to be perfect.'
        self._print_instruct(instruct)

        self._rough_circle, keypress = drag_circle(self._cal_image)

        # adjust procedure counter
        self._keypress_go_to(keypress)
        return

    def _get_final_circle(self):
        """Manually adjust a circle on an image"""

        instruct = 'Adjust the circle so that it is co-radial with the ' \
                   'tracking target.'
        self._print_instruct(instruct)

        self.circle, keypress = adjust_circle(self._cal_image,
                                              self._rough_circle)

        # adjust procedure counter
        self._keypress_go_to(keypress)
        return

    def _get_thresholds(self):
        """finds preliminary threshold values by analyzing pixels
        within selected circle and getting limits based on lower and
        upper percentile values.  Then displays thresholded HSV binary and
        allows user to adjust threshold limits.
        """

        instruct = 'Adjust the HSV threshold limits until the target is ' \
                   'highly visible and the rest of the image is mostly masked.'
        kp_reset = 'Press \'r\' to reset trackbar positions.'
        self._print_instruct(instruct, kp_before=kp_reset)

        self.thresholds, keypress = modify_thresholds(self._vid_feed,
                                                      self._cal_image,
                                                      self._rough_circle)

        # adjust procedure counter
        self._keypress_go_to(keypress)
        return

    def _keypress_go_to(self, keypress):
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

    def _print_instruct(self, message, kp_before=None):
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

        print('*' * 79)

        return

    def adjust_thresholds(self):
        """Displays thresholded HSV binary and allows user to adjust
        threshold limits."""

        instruct = 'Adjust the HSV threshold limits until the target is ' \
                   'highly visible and the rest of the image is mostly masked.'
        kp_reset = 'Press \'r\' to reset trackbar positions.'
        kp_accept = 'Press SPACE to accept adjusted values.'
        kp_abort = 'Press ESC or \'q\' to abort adjustment.'
        print('*' * 79, instruct, '', kp_reset, kp_accept, kp_abort, '*' * 79,
              sep='\n')

        vid_feed = cv2.VideoCapture(self._cam)
        adj_thresholds = None
        keypress = -1

        while keypress not in (27, 32, 113):
            adj_thresholds, keypress = tune_thresholds(vid_feed,
                                                       self.thresholds)

        if keypress == 32:  # SPACE
            self.thresholds = adj_thresholds
            print('SPACE pressed.  Threshold values adjusted.')
        elif keypress in (27, 113):  # ESC
            print('ESC or\'q\' pressed.  Adjustment aborted.')

        vid_feed.release()
        return


# noinspection PyUnusedLocal
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
        params['released'] = True
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


def get_threshold_limits(image, percentiles):
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


def circle_mask(image, circle, crop=False):
    """Create a binary mask where all points outside the input circle are
    set to (0, 0, 0).  If crop is True, returns only an area of the image
    around the circle, to reduce computations.
    """
    center = circle[0]
    radius = circle[1]

    mask = np.zeros_like(image)
    cv2.circle(mask, tuple(center), radius, (1, 1, 1), -1)
    masked = image * mask
    if crop:
        pad_radius = radius + 1
        # reversed because center is indexed (x, y) where image is (y, x)
        roi_min = reversed(np.subtract(center, pad_radius))
        roi_max = reversed(np.add(center, pad_radius))
        frame_min = [0, 0]
        frame_max = image.shape
        mins = [max(*v) for v in zip(roi_min, frame_min)]
        maxs = [min(*v) for v in zip(roi_max, frame_max)]
        masked = masked[mins[0]: maxs[0], mins[1]: maxs[1]]
    return masked


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


def focus_window(win):
    """Activate window(bring to front) by executing a shell command with the
    wmctrl package.  Only works on linux, must have wmctrl package.  If
    package is not available, function will print message but continue with
    program, and user will have to click on window to bring it forward.
    """
    if sys.platform == 'linux':
        try:
            # short delay to ensure window is initialized before attempting
            # to bring to front
            time.sleep(.01)
            call(['wmctrl', '-a', win])
        except FileNotFoundError:
            pass
    else:
        pass
    return


def capture_image(vid_feed):
    """Displays video from input feed and waits for a keypress to capture an
    image.  Returns captured image and keypress.
    """

    # initialize window
    win = 'Camera Feed'
    cv2.namedWindow(win)
    focus_window(win)

    # initialize variables
    image = None
    keypress = -1

    while keypress == -1:
        __, image = vid_feed.read()
        cv2.imshow(win, image)
        keypress = cv2.waitKey(10)

    cv2.destroyWindow(win)
    return image, keypress


def drag_circle(image):
    """Displays input image and lets user draw a circle by dragging the
    mouse from the center to the edge.  Waits for keypress to finish.
    Returns circle parameters(center, radius) and keypress.
    """

    # initialize window
    win = 'Select Calibration Region'
    cv2.namedWindow(win)

    # initialize variables
    cb_params = {'drag': None,
                 'point1': np.zeros(2),
                 'point2': np.zeros(2),
                 'released': False}
    center = None
    radius = None
    keypress = -1

    cv2.imshow(win, image)

    while keypress == -1:

        # mouse callback function for dragging circle on window
        cv2.setMouseCallback(win, circle_mouse_callback,
                             param=cb_params)

        center = cb_params['point1']
        radius = np.int(np.linalg.norm(cb_params['point1'] -
                                       cb_params['point2']))

        # continuously draw circle on image while mouse is being dragged
        if cb_params['drag'] or cb_params['released']:
            circ_img = image.copy()
            cv2.circle(circ_img,
                       tuple(center),
                       radius,
                       (0, 0, 0), 1, 8, 0)
            cv2.imshow(win, circ_img)

        keypress = cv2.waitKey(5)

    circle = [center, radius]

    cv2.destroyWindow(win)
    return circle, keypress


def adjust_circle(image, circle):
    """Manually adjust a circle on an image.  Takes an input image and
    circle(center, radius) and shows a blown up region centered around the
    circle.  Allows the user to adjust the circle using trackbars.  Waits
    for keypress to finish.  Returns adjusted circle and keypress."""

    # initialize window and trackbars
    win = 'Adjust Target Circle'
    cv2.namedWindow(win)
    cv2.resizeWindow(win, 200, 200)

    # initialize variables
    roi, roi_origin = get_circle_roi(image, circle)

    circle_local = np.copy(circle)
    circle_local[0] = circle[0] - np.flipud(roi_origin)

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

    adj_circle = ((cv2.getTrackbarPos('x', win) // scale +
                   roi_origin[1],
                   cv2.getTrackbarPos('y', win) // scale +
                   roi_origin[0]),
                  cv2.getTrackbarPos('r', win) // scale)

    cv2.destroyWindow(win)
    return adj_circle, keypress


def circle_thresh_limits(image, circle):
    """DOCSTRING"""
    masked_circle = circle_mask(image, circle, crop=True)
    thresholds = get_threshold_limits(masked_circle,
                                      config['thresh_percs'])
    return thresholds


def tune_thresholds(vid_feed, thresholds):
    """DOCSTRING"""
    # initialize window
    win = 'Adjustment Control Panel'
    cv2.namedWindow(win)

    # initialize variables and trackbars
    thresh_names = ['H_LO', 'H_HI',
                    'S_LO', 'S_HI',
                    'V_LO', 'V_HI']

    blur_k_name = 'BLUR \'K\''

    max_vals = [179, 179, 255, 255, 255, 255]

    for thresh, val, max_val in zip(thresh_names,
                                    thresholds,
                                    max_vals):
        cv2.createTrackbar(thresh, win, val, max_val, empty_callback)

    cv2.createTrackbar(blur_k_name, win,
                       config['blur_k']['initial'],
                       config['blur_k']['max'],
                       empty_callback)
    cv2.setTrackbarMin(blur_k_name, win, 1)

    thresh_vals = None
    keypress = -1

    while keypress == -1:
        __, frame = vid_feed.read()

        # blur frame
        blur_k = cv2.getTrackbarPos(blur_k_name, win)
        frame_blur = cv2.blur(frame, (blur_k, blur_k))

        frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        thresh_vals = np.asarray([cv2.getTrackbarPos(name, win)
                                  for name in thresh_names])

        frame_thresh = cv2.inRange(frame_hsv,
                                   thresh_vals[::2],
                                   thresh_vals[1::2])

        # cv2.imshow(win_thresh, frame_thresh)
        cv2.imshow(win, frame_thresh)

        keypress = cv2.waitKey(5)

    cv2.destroyWindow(win)
    vid_feed.release()
    return thresh_vals, keypress


def modify_thresholds(vid_feed, image, circle):
    """Takes input image and circle and analyzes all pixels within that circle
    to find threshold limits for HSV channels based on lower and upper
    percentile values.  Then uses thresholds to display a binary video feed,
    with trackbars that allow the user to adjust the threshold values(and
    blur 'k' value) in real time until they find values that are suitable.
    Waits for keypress to finish.  Returns adjusted threshold values and
    keypress.
    """

    thresholds = circle_thresh_limits(image, circle)
    adj_thresholds, keypress = tune_thresholds(vid_feed, thresholds)

    return adj_thresholds, keypress


# TESTING
if __name__ == "__main__":
    cam = config['cameras'][0]
    target = Target(cam)
    target.calibrate()
    target.adjust_thresholds()
