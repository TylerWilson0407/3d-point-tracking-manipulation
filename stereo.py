"""DOCSTRING"""
from config import config as cf
import cv2
from imutils.video import WebcamVideoStream
import numpy as np
from subprocess import call
import sys
import time


def chessboard_cap(cam_ids, chessboard, num_frames):
    """Capture frames from input cameras containing a chessboard calibration
    rig in order to calibrate the intrinsic and extrinsic parameters of the
    camera setup.
    """

    stream_prefix = 'Camera'
    cap_prefix = 'Capture'

    streams = initialize_streams(cam_ids)
    stream_windows = initialize_windows(stream_prefix, cam_ids)
    cap_windows = initialize_windows(cap_prefix, cam_ids)

    focus_window(stream_windows[0])

    frames = []
    corner_list = []

    while len(frames) < num_frames:

        capture_instruct(len(frames), num_frames)
        images, keypress = capture_multi(streams, stream_windows)
        retvals, corners = find_corners(images, cap_windows,
                                        tuple(chessboard['dims']))

        # clean this up???
        if keypress == 32:  # SPACE
            if all(retvals):
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

    print('*' * 79)
    print('{0} frames captured.'.format(num_frames))
    print('*' * 79)

    return corner_list


def initialize_streams(cam_ids):
    """Initialize webcam streams on input cameras.  Uses multithreading to
    prevent camera desync due to camera buffers."""

    streams = []
    for cam in cam_ids:
        streams.append(WebcamVideoStream(cam).start())

    return streams


def initialize_windows(prefix, cam_ids):
    """Initialize windows for camera video feeds."""

    windows = []
    for cam in cam_ids:
        win_name = prefix + ' ' + str(cam)
        windows.append(win_name)
        cv2.namedWindow(win_name)

    return windows


def capture_multi(streams, windows):
    """Displays videos from multiple input feeds and waits for a keypress to
    capture images.  Returns captured images and keypress.
    """

    # initialize variables
    images = [None] * len(streams)
    keypress = -1

    while keypress == -1:

        for i, (stream, win) in enumerate(zip(streams, windows)):
            images[i] = stream.read()
            cv2.imshow(win, images[i])
        keypress = cv2.waitKey(5)

    return images, keypress


def capture_instruct(current_frame, total_frames):
    """Print instructions for capturing frames."""

    print('*' * 79)
    print('{0}/{1} frames captured.'.format(current_frame, total_frames))
    print('')
    print('Press SPACE to capture.')
    print('Press ESC to go back one frame.')
    print('Press \'q\' to quit.')

    return


def find_corners(images, windows, chess_dims):
    """Display chessboard corners on input images."""

    retvals = [False] * len(images)
    corners = [None] * len(images)

    for i, (image, window) in enumerate(zip(images, windows)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retvals[i], corners[i] = cv2.findChessboardCorners(image, chess_dims)
        cv2.drawChessboardCorners(image, chess_dims, corners[i], 0)
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


def calibrate_cams(chessboard, corner_list):
    """Calibrate cameras using chessboard corner data.  Determines intrinsic
    and extrinsic parameters of stereo camera configuration.
    """

    pass


if __name__ == "__main__":
    chessboard_cap(cf['cameras'], cf['chessboard'], cf['stereo_cal_frames'])
