"""DOCSTRING"""
from config import config
import cv2
import numpy as np
import time


def Class():
    def __init__(self):
        pass


def chessboard_cap(cam_ids):
    """Capture frames from input cameras containing a chessboard calibration
    rig in order to calibrate the intrinsic and extrinsic parameters of the
    camera setup.
    """

    vid_feeds = []
    windows = []
    for i, cam in enumerate(cam_ids):
        vid_feeds.append(cv2.VideoCapture(cam))
        windows.append('Camera ' + str(i))
        cv2.namedWindow(windows[i])


    chess = config['chessboard']
    chess_dims = (chess['hor_sq'], chess['ver_sq'])
    frames = config['stereo_calib']['num_frames']

    capture_multi(vid_feeds, windows)

    cv2.waitKey(0)


def capture_multi(vid_feeds, windows):
    """Displays videos from multiple input feeds and waits for a keypress to
    capture images.  Returns captured images and keypress.
    """

    # initialize variables
    images = [None] * len(vid_feeds)
    keypress = -1

    while keypress == -1:

        for feed, image, win in zip(vid_feeds, images, windows):
            __, image = feed.read()
            cv2.imshow(win, image)
        keypress = cv2.waitKey(5)

    return images, keypress


if __name__ == "__main__":
    cam_ids = config['cameras']
    chessboard_cap(cam_ids)
