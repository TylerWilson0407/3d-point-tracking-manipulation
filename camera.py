# Remove this later if merging TrackingTarget into calibration module
from target import Target
from config import config
import cv2
import numpy as np
from subprocess import call
import sys
import time


class Camera:
    def __init__(self, cam_id):
        self._cam = cam_id

        # names for targets - short for index, middle, thumb(fingers)
        targ_names = config['targets']
        self.targets = create_tracking_targets(targ_names, cam_id)

        self._cal_count = None
        self._cal_funcs = [self._get_cal_frame,
                           self._select_target]

        self._cal_image = None
        self._circle_draw = None

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

        vid_feed = cv2.VideoCapture(self._cam)

        self._cal_image, keypress = capture_image(vid_feed)

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


def target_color(thresholds):
    """Get average HSV values from input thresholds and convert to BGR."""
    color = [(a + b) // 2 for a, b in zip(thresholds[:5:2],
                                          thresholds[1::2])]

    # color[1:] = [255, 255]
    col_array = np.asarray(color).reshape([1, 1, 3]).astype('uint8')
    color_rgb = cv2.cvtColor(col_array, cv2.COLOR_HSV2BGR)
    color_rgb = color_rgb[0][0].tolist()
    return color_rgb


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
            return
        except FileNotFoundError:
            pass
    else:
        pass
    return


def capture_image(vid_feed):
    """Displays video from input feed and waits for a keypress.  Returns
    captured image and keypress.
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


# TESTING
if __name__ == "__main__":
    test_cam = Camera(0)
    test_cam.calibrate_targets()
