"""DOCSTRING"""
from config import config
import cv2
from imutils.video import WebcamVideoStream
import numpy as np
import time


class Chessboard:
    def __init__(self, hor_sq, ver_sq, num_frames):
        self.dims = (hor_sq, ver_sq)
        self.num_frames = num_frames


def chessboard_cap(cam_ids, chessboard):
    """Capture frames from input cameras containing a chessboard calibration
    rig in order to calibrate the intrinsic and extrinsic parameters of the
    camera setup.
    """

    streams = initialize_streams(cam_ids)
    windows = initialize_windows(cam_ids)

    frames = []

    while len(frames) > chessboard.num_frames:

        images, keypress = capture_multi(streams, windows)



    return


def initialize_streams(cam_ids):
    """Initialize webcam streams on input cameras.  Uses multithreading to
    prevent camera desync due to camera buffers."""

    streams = []
    for cam in cam_ids:
        streams.append(WebcamVideoStream(cam).start())

    return streams


def initialize_windows(cam_ids):
    """Initialize windows for camera video feeds."""

    windows = []
    for cam in cam_ids:
        win_name = 'Camera ' + str(cam)
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

        for stream, image, win in zip(streams, images, windows):
            image = stream.read()
            cv2.imshow(win, image)
        keypress = cv2.waitKey(5)

    return images, keypress


if __name__ == "__main__":
    cams = config['cameras']
    chessboard_cap(cams)
