"""DOCSTRING"""
from config import config as cf
import cv2
from imutils.video import WebcamVideoStream
import numpy as np
import os
import pickle
from subprocess import call
import sys
import time
# below are for testing, remove later
import math
import matplotlib.pyplot as plt


class Camera:
    """DOCSTRING"""

    def __init__(self, cam_id):
        self.id = cam_id
        self.stream = WebcamVideoStream(self.id).start()
        self.size = get_image_size(self)
        self.cam_mat = None
        self.dist_coeff = None

        # targets
        # names for targets - short for index, middle, thumb(fingers)
        targ_names = cf['targets']
        self.targets = create_tracking_targets(targ_names, cam_id)

        self._cal_count = None
        self._cal_funcs = [self._get_cal_frame,
                           self._select_target]

        self._cal_image = None
        self._circle_draw = None

    def calibrate_intrinsic(self, chessboard, frame_count):

        image_points = chessboard_cap(self, chessboard, frame_count)

        object_pts = chessboard.expand_points(image_points)

        self.cam_mat, self.dist_coeff = calibrate_intrinsic(object_pts,
                                                            image_points,
                                                            self.size)

        show_undistorted(self)

        return

    def calibrate_targets(self):
        """EXPAND HERE when you figure out how to layout the entire
        calbration procedure(e.g. break up into camera and target
        calibration functions?).
        """

        self._cal_count = 0

        self._cal_image = None

        start_count = 0

        cal_check = False
        self._circle_draw = np.empty(0)

        # loop through until all targets are calibrated
        # any way to clean up this mess???
        while (not cal_check) and (start_count <= self._cal_count):
            self._cal_funcs[self._cal_count]()

            cal_check = all([getattr(self.targets[k], 'calibrated')
                             for k in self.targets])

            self.draw_calib_circle()

    def draw_calib_circle(self):
        """Iterate through targets and draw circle over calibrated targets."""

        if not self._circle_draw.any():
            self._circle_draw = self._cal_image.copy()
        for target in self.targets.values():
            if target.calibrated:
                cv2.circle(self._circle_draw,
                           target.circle[0],
                           target.circle[1],
                           target_color(target.thresholds),
                           -1)
        return

    def _get_cal_frame(self):
        """Capture calibration image from input video feed"""

        instruct = 'Hold up calibration target(s) and press SPACE to ' \
                   'capture image.'
        self._print_instruct(instruct)

        vid_feed = cv2.VideoCapture(self.id)

        # self._cal_image, keypress = capture_targets(vid_feed)
        self._cal_image, keypress = capture_image(self, 'Camera Feed')

        # adjust procedure counter
        self._keypress_go_to(keypress)
        return

    def _select_target(self):
        """DOCSTRING"""
        # print('PLACEHOLDER: i, m, or t')

        select_target_message(self.targets)

        win = 'Choose Calibration Target'
        cv2.namedWindow(win)

        cv2.imshow(win, self._circle_draw)

        keypress = -1

        while keypress == -1:
            keypress = cv2.waitKey(5)

        cv2.destroyWindow(win)

        # make this better when you have time
        if keypress == 27:  # ESC
            self._cal_count -= 1
        elif keypress == 113:  # 'q'
            self._cal_count = -1
        else:
            for key in self.targets.keys():
                if chr(keypress) == key[0]:
                    print('\'{0}\' target selected.'.format(key))
                    self.targets[key].calibrate_intrinsic(self._cal_image)
        return

    def _print_instruct(self, message, kp_before=None, kp_after=None):
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


class Chessboard:
    """DOCSTRING"""

    def __init__(self, spacing, dims):
        self.spacing = spacing
        self.dims = tuple(dims)

    @property
    def points(self):
        """ Builds matrix of 3D points with locations of chessboard corners.
        Z-coordinates are zero.
        """

        # initialize (H*V) x 3 matrix of points
        points = np.zeros((self.dims[0] * self.dims[1], 3), np.float32)

        # populate x,y coordinates with 2D corner locations and reshape
        points[:, :2] = self.spacing * (np.mgrid[0:self.dims[0],
                                        0:self.dims[1]].T.reshape(-1, 2))

        return points

    def expand_points(self, count):
        """Multiplies points matrix by count.  If count is an integer,
        multiplies by that integer, if it is a list or numpy array,
        multiplies by the length of that list/array.
        """

        if type(count) is int:
            points_list = [self.points] * count
        elif (type(count) is list) or (type(count) is np.ndarray):
            points_list = [self.points] * len(count)
        else:
            raise TypeError("Input must be integer, list, or numpy array.")

        return points_list


class Stereo:
    """DOCSTRING"""

    def __init__(self, cameras, chessboard, frame_count):
        self.cameras = cameras
        self.chessboard = chessboard
        self.frame_count = frame_count

        self.rot_mat = None
        self.trans_vec = None
        self.rect_mat = None
        self.proj_mat = None

        self.corner_list = None

    def load_calib_data(self, filename):
        """If calibration file exists, loads saved calibration data and inputs
            into stereo object.
            """

        if os.path.exists(filename):
            data = pickle_load(filename)

            for camera in stereo.cameras:
                camera.cam_mat = data['cameras'][camera.id]['cam_mat']
                camera.cam_mat = data['cameras'][camera.id]['dist_coeff']

            stereo.rot_mat = data['rot_mat']
            stereo.trans_vec = data['trans_vec']
            stereo.rect_mat = data['rect_mat']
            stereo.proj_mat = data['proj_mat']
        else:
            print('Calibration file not found.  Recalibrate all.')

        return


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

        # check if image is input as a parameter and skip image capture if so
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

        self._cal_image, keypress = capture_targets(self._vid_feed)

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
        upper percentile values.  Then displays threshold HSV binary and
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
    scale = cf['roi']['ADJUST_SCALE']
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


def calibrate_intrinsic(object_points, image_points, image_size):
    """Calibrate camera intrinsic pattern from input object points(real
    coordinates of points on chessboard calibration rig) and input image
    points(points on the chessboard as seen by camera).
    """

    __, cam_mat, dist_coeff, __, __ = cv2.calibrateCamera(object_points,
                                                          image_points,
                                                          image_size, 0, 0)

    return cam_mat, dist_coeff


def calibrate_stereo(cameras, chessboard, corner_list):
    """DOCSTRING
    NOTE - only works with two cameras at this time, may change later
    """

    real_pts = chessboard.expand_points(corner_list[0])

    retval, __, __, __, __, rot_mat, trans_vec, ess_mat, fund_mat \
        = cv2.stereoCalibrate(real_pts,
                              corner_list[0],
                              corner_list[1],
                              cameras[0].cam_mat,
                              cameras[0].dist_coeff,
                              cameras[1].cam_mat,
                              cameras[1].dist_coeff,
                              cameras[0].size,
                              flags=cv2.CALIB_FIX_INTRINSIC)

    print(retval)

    return rot_mat, trans_vec, ess_mat, fund_mat


#  UNUSED?
# def capture_chessboard_images(cam_ids, chessboard, num_frames):
#     """Capture frames from input cameras of chessboard calibration rig."""
#
#     stream_prefix = 'Camera'
#     cap_prefix = 'Capture'
#
#     streams = initialize_streams(cam_ids)
#     stream_windows = initialize_windows(stream_prefix, cam_ids)
#     cap_windows = initialize_windows(cap_prefix, cam_ids)
#
#     focus_window(stream_windows[0])
#
#     frames = []
#
#     while len(frames) < num_frames:
#
#         capture_instruct(len(frames), num_frames)
#         images, keypress = capture_image(streams, stream_windows)
#         retvals, corners = find_corners(images, cap_windows,
#                                         chessboard.dims)
#
#         # clean this up???
#         if keypress == 32:  # SPACE
#             if all(retvals):
#                 frames.append(images)
#             else:
#                 print('*' * 79)
#                 print('Not all corners found, recapture.')
#         elif keypress == 27:  # ESC
#             if len(frames) > 0:
#                 frames.pop()
#             else:
#                 print('Aborting.')
#                 return
#         elif keypress == 113:  # 'q'
#             print('Aborting.')
#             return
#
#     return frames


def capture_image(cameras, windows):
    """Displays videos from multiple input feeds and waits for a keypress to
    capture images.  Returns captured images and keypress.
    """

    # initialize variables
    images = [None] * len(cameras)
    keypress = -1

    while keypress == -1:

        for i, (camera, win) in enumerate(zip(cameras, windows)):
            images[i] = camera.stream.read()
            cv2.imshow(win, images[i])
        keypress = cv2.waitKey(5)

    return images, keypress


def capture_instruct(current_frame, total_frames):
    """Print instructions for capturing frames."""

    print('*' * 79)
    print('{0}/{1} frames captured.'.format(current_frame, total_frames))
    print('')
    if current_frame < total_frames:
        print('Press SPACE to capture.')
    else:
        print('Press SPACE to finish capturing frames.')
    print('Press ESC to go back one frame.')
    print('Press \'q\' to quit.')

    return


def capture_targets(vid_feed):
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


def check_create_file(filename):
    """Checks if the file exists.  If not, creates file containing empty
    dictionary.
    """

    if os.path.exists(filename):
        return
    else:
        data = {}
        pickle_dump(filename, data)

    return


def chessboard_cap(cameras, chessboard, num_frames):
    """Capture frames from input cameras containing a chessboard calibration
    rig in order to calibrate the intrinsic and extrinsic parameters of the
    camera setup.  Returns pixel locations of captured corners as well as a
    list of images sizes of cameras(needed for camera calibration).
    """

    cam_list = to_list(cameras)

    stream_prefix = 'Camera'
    cap_prefix = 'Capture'

    stream_windows = initialize_windows(stream_prefix, cam_list)
    cap_windows = initialize_windows(cap_prefix, cam_list)

    focus_window(stream_windows[0])

    frames = []
    corner_list = []

    while len(frames) <= num_frames:

        capture_instruct(len(frames), num_frames)
        images, keypress = capture_image(cam_list, stream_windows)
        retvals, corners = find_corners(images, cap_windows,
                                        chessboard)

        # clean this up???
        if keypress == 32:  # SPACE
            if len(frames) == num_frames:
                break
            elif all(retvals):
                frames.append(images)
                corner_list.append(corners)
            else:
                print('*' * 79)
                print('Not all corners found, recapture.')
        elif keypress == 27:  # ESC
            if len(frames) > 0:
                frames.pop()
            else:
                print('Aborting.')
                return
        elif keypress == 113:  # 'q'
            print('Aborting.')
            return

    # reshape corner_list where first index is the camera containing list of
    #  frames, vs first index being frames with lists for each camera
    corner_list = np.reshape(corner_list, (len(cam_list),
                                           num_frames,
                                           chessboard.dims[0] *
                                           chessboard.dims[1],
                                           1, 2))

    print('*' * 79)
    print('{0} frames captured.'.format(num_frames))
    print('*' * 79)

    frame_sizes = []

    for frame in frames[0]:
        size = [frame.shape[0], frame.shape[1]]
        frame_sizes.append(size)

    cv2.destroyAllWindows()

    # if only one camera input, extract from nested list
    # (should this structure be changed?)
    if len(cam_list) == 1:
        corner_list = corner_list[0]

    return corner_list


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


def circle_thresh_limits(image, circle):
    """DOCSTRING"""
    masked_circle = circle_mask(image, circle, crop=True)
    thresholds = get_threshold_limits(masked_circle,
                                      cf['thresh_percs'])
    return thresholds


def create_cam_dict(cam_ids):
    """Output dictionary with indexed keys(cam0, cam1, etc) with values of
    the camera IDs.
    """
    cams = {}
    for i, cam in enumerate(cam_ids):
        key = '_cam' + str(i)
        cams[key] = cam
    return cams


def create_tracking_targets(keys, cam):
    """Output dictionary with keys as keys and TrackingTarget objects as
    values."""

    targets = {}
    for key in keys:
        targets[key] = Target(cam)

    return targets


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


def empty_callback(*_, **__):
    """Empty callback function for passing to functions that require a
    callback.  Accepts any argument and returns None.
    """
    return None


def find_corners(images, windows, chessboard):
    """Display chessboard corners on input images."""

    retvals = [[False] for __ in range(len(images))]
    corners = [[None] for __ in range(len(images))]

    for i, (image, window) in enumerate(zip(images, windows)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retvals[i], corners[i] = cv2.findChessboardCorners(image,
                                                           chessboard.dims)
        cv2.drawChessboardCorners(image,
                                  chessboard.dims,
                                  corners[i],
                                  retvals[i])
        cv2.imshow(window, image)

    return retvals, corners


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


def get_circle_roi(image, circle):
    """Get a region of interest around a defined circle in an image."""
    center = circle[0]
    radius = circle[1]
    padding = cf['roi']['PADDING']
    rad_padded = int(radius * padding)

    y_min = max((center[1] - rad_padded), 0)
    y_max = min((center[1] + rad_padded), image.shape[0])
    x_min = max((center[0] - rad_padded), 0)
    x_max = min((center[0] + rad_padded), image.shape[1])

    roi = image[y_min: y_max, x_min: x_max]

    roi_origin = np.array([y_min, x_min])

    return roi, roi_origin


def get_image_size(camera):
    """Get image size of input camera ID"""

    frame = camera.stream.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame.shape


def get_rect_matrices(cameras, rot_mat, trans_vec):
    """DOCSTRING"""

    rect = [np.empty((3, 3)) for __ in range(len(cameras))]
    proj = [np.empty((3, 4)) for __ in range(len(cameras))]

    rect[0], rect[1], proj[0], proj[1], __, __, __ = \
        cv2.stereoRectify(cameras[0].cam_mat,
                          cameras[0].dist_coeff,
                          cameras[1].cam_mat,
                          cameras[1].dist_coeff,
                          cameras[0].size,
                          rot_mat,
                          trans_vec)

    return rect, proj


def get_rect_maps(cameras, rect, proj):
    """Computes rectification maps."""

    # empty_mat = np.empty(tuple(frame_size))
    # map1 = [empty_mat for __ in range(len(cam_mats))]
    # map2 = [empty_mat for __ in range(len(cam_mats))]

    map1 = [np.empty(cameras[0].size) for __ in range(len(cameras))]
    map2 = [np.empty(cameras[0].size) for __ in range(len(cameras))]

    for i, (camera, r, p) in enumerate(zip(cameras, rect, proj)):
        map1[i], map2[i] = cv2.initUndistortRectifyMap(camera.cam_mat,
                                                       camera.dist_coeff,
                                                       r,
                                                       p,
                                                       camera.size,
                                                       cv2.CV_32FC1)

    return map1, map2


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


def initialize_cameras(cam_ids):
    """DOCSTRING"""

    cameras = []

    for cam_id in cam_ids:
        cameras.append(Camera(cam_id))

    return cameras


def initialize_windows(prefix, cameras):
    """Initialize windows for camera video feeds."""

    windows = []
    for cam in cameras:
        win_name = prefix + ' ' + str(cam.id)
        windows.append(win_name)
        cv2.namedWindow(win_name)

    return windows


def kill_streams(streams):
    """Kill stream threads."""

    for stream in streams:
        stream.stop()

    return


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


def pickle_dump(filename, data):

    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

    return


def pickle_load(filename):

    with open(filename, 'rb') as fp:
        data = pickle.load(fp)

    return data


def save_intrinsic(cameras, filename):
    """Saves intrinsic camera data to calibration file."""

    check_create_file(filename)

    data = pickle_load(filename)

    for camera in cameras:
        data['cameras'][camera.id] = {'cam_mat': camera.cam_mat,
                                      'dist_coeff': camera.dist_coeff}

    pickle_dump(filename, data)

    return


def scale_image(image, scale):
    """Resize an image by a given scale."""
    x_scale = int(np.around(image.shape[1] * scale))
    y_scale = int(np.around(image.shape[0] * scale))
    scaled_image = cv2.resize(image, (x_scale, y_scale))
    return scaled_image


def select_target_message(targets):
    """Print message for select_target method."""
    star_line = '*' * 79
    message = 'Choose target to calibrate.'

    kp_esc = 'Press ESC to return to previous step.'
    kp_q = 'Press \'q\' to abort calibration procedure.'

    print(star_line, message, '', sep='\n')

    for k, v in targets.items():
        if not v.calibrated:
            print('\'{0}\' target NOT CALIBRATED.  Press \'{1}\' to '
                  'calibrate.'.format(k, k[0]))
        else:
            print('\'{0}\' target CALIBRATED.  Press \'{1}\' to '
                  'recalibrate.'.format(k, k[0]))

    print(kp_esc, kp_q, star_line, sep='\n')


def show_image(images, windows):
    """Displays videos from multiple input feeds and waits for a keypress to
    capture images.  Returns captured images and keypress.
    """

    print('*' * 79)
    print('Displaying images.  Press any key to continue.')
    print('*' * 79)

    # initialize variables
    keypress = -1

    while keypress == -1:

        for image, win in zip(images, windows):
            cv2.imshow(win, image)
        keypress = cv2.waitKey(5)

    return


def show_rectified(cameras, rect, proj, map1, map2):
    """Computes undistortion and rectification maps and remaps camera
    outputs to show a stereo undistorted image."""

    udists = [np.empty(cameras[0].size) for __ in range(len(cameras))]

    stream_prefix = 'Camera'

    stream_windows = initialize_windows(stream_prefix, cameras)

    focus_window(stream_windows[0])

    keypress = -1

    while keypress == -1:

        for i, (camera, m1, m2) in enumerate(zip(cameras, map1, map2)):
            cap = camera.stream.read()
            cap = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
            cv2.imshow(stream_windows[i], cap)

            cv2.remap(cap, m1, m2, cv2.INTER_LINEAR, udists[i])

        udist_join = np.zeros((cameras[0].size[0],
                               cameras[0].size[1] * len(udists)), np.uint8)

        x_orig = 0

        for udist in udists:
            udist_join[:udist.shape[0], x_orig:x_orig + udist.shape[1]] = udist
            x_orig += udist.shape[1]

        cv2.imshow('Stereo Images', udist_join)
        # cv2.imshow('Stereo Images 0', udists[0])
        # cv2.imshow('Stereo Images 1', udists[1])

        keypress = cv2.waitKey(5)

    return


def show_remap(cameras, map1, map2):
    """Computes undistortion and rectification maps and remaps camera
    outputs to show a stereo undistorted image."""

    stream_prefix = 'Camera'

    stream_windows = initialize_windows(stream_prefix, cameras)
    dst_windows = initialize_windows('dst', cameras)

    focus_window(stream_windows[0])

    keypress = -1

    while keypress == -1:

        for cam, swin, dwin, m1, m2 in zip(cameras,
                                              stream_windows,
                                              dst_windows,
                                              map1,
                                              map2):
            cap = cam.stream.read()
            cap = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

            dst = cv2.remap(cap, m1, m2, cv2.INTER_LINEAR)

            cv2.imshow(swin, cap)
            cv2.imshow(dwin, dst)

        keypress = cv2.waitKey(5)

    return


def show_undistorted(cameras):
    """Display undistorted image(s)."""

    cam_list = to_list(cameras)

    stream_prefix = 'Camera'

    stream_windows = initialize_windows(stream_prefix, cam_list)

    focus_window(stream_windows[0])

    keypress = -1

    while keypress == -1:

        for win, camera in zip(stream_windows, cam_list):
            image = camera.stream.read()

            dst = cv2.undistort(image, camera.cam_mat, camera.dist_coeff)

            cv2.imshow(win, dst)

        keypress = cv2.waitKey(5)

    cv2.destroyAllWindows()

    return


def target_color(thresholds):
    """Get average HSV values from input thresholds and convert to BGR."""
    color = [(a + b) // 2 for a, b in zip(thresholds[:5:2],
                                          thresholds[1::2])]

    # color[1:] = [255, 255]
    col_array = np.asarray(color).reshape([1, 1, 3]).astype('uint8')
    color_rgb = cv2.cvtColor(col_array, cv2.COLOR_HSV2BGR)
    color_rgb = color_rgb[0][0].tolist()
    return color_rgb


def to_list(element):
    """If input is not a list, converts to single element list."""

    if type(element) is not list:
        return [element]
    else:
        return element


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
                       cf['blur_k']['initial'],
                       cf['blur_k']['max'],
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


# TESTING FUNCTIONS - delete later
def test_calib_intrinsic_save():

    cameras = initialize_cameras(cf['cam_ids'])

    chessboard = Chessboard(cf['chessboard']['spacing'],
                            cf['chessboard']['dims'])

    for camera in cameras:
        camera.calibrate_intrinsic(chessboard, cf['calib_frame_count']['intrinsic'])

    # intrinsic_data = {'cameras': cameras}

    intrinsic_data = {'cam_mats': [camera.cam_mat for camera in cameras],
                      'dist_coeffs': [camera.dist_coeff for camera in cameras]}

    with open('intrinsic_data', 'wb') as fp:
        pickle.dump(intrinsic_data, fp)

    return


def test_cornercap_save(cameras):
    """testing function"""

    chessboard = Chessboard(cf['chessboard']['spacing'],
                            cf['chessboard']['dims'])

    stereo = Stereo(cameras, chessboard, cf['calib_frame_count']['stereo'])

    corner_list = chessboard_cap(cameras,
                                 chessboard,
                                 stereo.frame_count)

    chess_cap_data = {'cameras': cameras,
                      'chessboard': chessboard,
                      'corner_list': corner_list}

    with open('chesscapdata', 'wb') as fp:
        pickle.dump(chess_cap_data, fp)

    return


def test_open_corner_data():
    """testing function"""

    with open('chesscapdata', 'rb') as fp:
        chess_cap_data = pickle.load(fp)

    cameras = chess_cap_data['cameras']
    chessboard = chess_cap_data['chessboard']
    corner_list = chess_cap_data['corner_list']


    return cameras, chessboard, corner_list


def test_open_saved(file, key):

    with open(file, 'rb') as fp:
        data = pickle.load(fp)

    return data[key]


def test_rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def test_compare_pts(chessboard, corner_list):

    real_pts = chessboard.points
    print(real_pts[:, 0])
    print(corner_list[0][0][:, :, 0].T[0])

    plt.plot(real_pts[:, 0], real_pts[:, 1], 'ro')
    plt.plot(corner_list[0][0][:, :, 0].T[0],
             corner_list[0][0][:, :, 1].T[0], 'bo')
    plt.plot(corner_list[1][0][:, :, 0].T[0],
             corner_list[1][0][:, :, 1].T[0], 'go')
    plt.show()

    return


# if __name__ == "__main__":
#
#     calib_intrinsic = True
#
#     if calib_intrinsic:
#         test_calib_intrinsic_save()
#
#     cameras = test_open_saved('intrinsic_data', 'cameras')
#
#     for camera in cameras:
#         camera.restart_stream()
#
#     for camera in cameras:
#         print(camera.cam_mat)
#         print(camera.dist_coeff)
#
#     cap_frames = True
#
#     if cap_frames:
#         test_cornercap_save(cameras)
#
#     cameras, chessboard, corner_list = test_open_corner_data()
#
#     # test_compare_pts(chessboard, corner_list)
#
#     r, t, e, f = calibrate_stereo(cameras, chessboard, corner_list)
#
#     print('r \n', r)
#     print('t \n', t)
#
#     #  testing reassignment
#     # r = np.eye(3)
#     # t = np.asarray([-120.0, 0.0, 0.0]).T
#
#     # show_undistorted(cameras)
#     #
#     rect, proj = get_rect_matrices(cameras, r, t)
#
#     print('rect \n', rect[0], '\n', rect[1])
#     print('proj \n', proj[0], '\n', proj[1])
#
#     map1, map2 = get_rect_maps(cameras, rect, proj)
#
#     show_rectified(cameras, rect, proj, map1, map2)
#
#     print(cameras[0].size)
#     print(map1[0].size)
#     print(map2[0].size)
#
#     show_remap(cameras, map1, map2)


if __name__ == "__main__":

    data_filename = cf['calib_data_filename']

    cameras = initialize_cameras(cf['cam_ids'])

    chessboard = Chessboard(cf['chessboard']['spacing'],
                            cf['chessboard']['dims'])

    stereo = Stereo(cameras, chessboard, cf['calib_frame_count']['stereo'])

    stereo.load_calib_data(data_filename)

    # test
    for camera in stereo.cameras:
        camera.calibrate_intrinsic(stereo.chessboard,
                                   cf['calib_frame_count']['intrinsic'])
