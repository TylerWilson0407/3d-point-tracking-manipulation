"""DOCSTRING"""
from config import config as cf
import cv2
from imutils.video import WebcamVideoStream
import numpy as np
import pickle
from subprocess import call
import sys
import time


def chessboard_cap(cam_ids, chessboard, num_frames):
    """Capture frames from input cameras containing a chessboard calibration
    rig in order to calibrate the intrinsic and extrinsic parameters of the
    camera setup.  Returns pixel locations of captured corners as well as a
    list of images sizes of cameras(needed for camera calibration).
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

    # reshape corner_list where first index is the camera containing list of
    #  frames, vs first index being frames with lists for each camera
    corner_list = np.reshape(corner_list, (len(cam_ids),
                                           num_frames,
                                           chessboard['dims'][0] *
                                           chessboard['dims'][1],
                                           1, 2))

    print('*' * 79)
    print('{0} frames captured.'.format(num_frames))
    print('*' * 79)

    frame_sizes = []

    for frame in frames[0]:
        size = [frame.shape[0], frame.shape[1]]
        frame_sizes.append(size)

    return corner_list, frame_sizes


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


def calibrate_cams(chessboard, corner_list, frame_sizes):
    """Calibrate cameras using chessboard corner data.  Determines intrinsic
    and extrinsic parameters of stereo camera configuration.
    """

    real_pts = [chessboard_points(chessboard)] * len(corner_list[0])

    cam_mats = []
    dist_coeffs = []

    for corners, frame_size in zip(corner_list, frame_sizes):

        __, cam_mat, dist_coeff, __, __ = \
            cv2.calibrateCamera(real_pts,
                                corners,
                                tuple(frame_size), 0, 0)

        cam_mats.append(cam_mat)
        dist_coeffs.append(dist_coeff)

    # cv2.stereoCalibrate function only estimates transformations between
    # two cameras.  To estimate transformations between subsequent cameras,
    # more stereo calibrations would be needed(i.e. calibrate 1 -> 2,
    # calibrate 2 -> 3, etc).  Will maybe add this functionality later.

    __, cam_mats[0], dist_coeffs[0], cam_mats[1], dist_coeffs[1], rot_mat, \
        trans_mat, ess_mat, fund_mat \
        = cv2.stereoCalibrate(real_pts,
                              corner_list[0],
                              corner_list[1],
                              cam_mats[0],
                              dist_coeffs[0],
                              cam_mats[1],
                              dist_coeffs[1],
                              tuple(frame_sizes[0]),
                              flags=cv2.CALIB_FIX_INTRINSIC)

    rect = [None] * 2
    proj = [None] * 2

    [rect[0], rect[1], proj[0], proj[1]] = \
        cv2.stereoRectify(cam_mats[0],
                          dist_coeffs[0],
                          cam_mats[1],
                          dist_coeffs[1],
                          tuple(frame_sizes[0]),
                          rot_mat,
                          trans_mat)[:4]

    return cam_mats, dist_coeffs, rect, proj


def chessboard_points(chessboard):
    """ Builds matrix of 3D points with locations of chessboard corners.
    Z-coordinates are zero.
    """

    dist = chessboard['dist']
    dims = chessboard['dims']

    # initialize (H*V) x 3 matrix of points
    points = np.zeros((dims[0] * dims[1], 3), np.float32)

    # populate x,y coordinates with 2D corner locations
    points[:, :2] = dist * np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1, 2)

    return points

def show_undistorted(cam_ids, cam_mats, dist_coeffs, rect, proj, frame_sizes):
    """Computes undistortion and rectification maps and remaps camera
    outputs to show a stereo undistorted image."""

    map1 = map2 = udist = [None] * len(cam_mats)

    for i, (cam, dist, r, p, size) in enumerate(zip(cam_mats,
                                                    dist_coeffs,
                                                    rect,
                                                    proj,
                                                    frame_sizes)):
        map1[i], map2[i] = cv2.initUndistortRectifyMap(cam,
                                                       dist,
                                                       r,
                                                       p,
                                                       tuple( size),
                                                       cv2.CV_32FC1)

    stream_prefix = 'Camera'

    streams = initialize_streams(cam_ids)
    stream_windows = initialize_windows(stream_prefix, cam_ids)

    focus_window(stream_windows[0])

    keypress = -1

    while keypress == -1:

        for i, (stream, m1, m2) in enumerate(zip(streams, map1, map2)):
            image = stream.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow(stream_windows[i], image)

            udist[i] = image
            cv2.remap(image, m1, m2, cv2.INTER_LINEAR, udist[i])

        udist_join = np.zeros((max(image.shape[0] for image in udist),
                              sum(image.shape[1] for image in udist)),
                              np.uint8)

        x_orig = 0

        for image in udist:
            udist_join[:image.shape[0], x_orig:x_orig + image.shape[1]] = image
            x_orig += image.shape[1]

        # print(udist_join)

        cv2.imshow('Stereo Images', udist_join)

        keypress = cv2.waitKey(5)

    return


if __name__ == "__main__":
    # corner_list, frame_sizes = chessboard_cap(cf['cameras'],
    #                                           cf['chessboard'],
    #                                           cf['stereo_cal_frames'])
    #
    # chess_cap_data = {'corners': corner_list,
    #                   'frame_size': frame_sizes}
    #
    # with open('chesscapdata', 'wb') as fp:
    #     pickle.dump(chess_cap_data, fp)

    with open('chesscapdata', 'rb') as fp:
        chess_cap_data = pickle.load(fp)

    # print(chess_cap_data)

    # points = chessboard_points(cf['chessboard'])
    cc = chess_cap_data['corners']
    fs = chess_cap_data['frame_size']

    cam_mats, dist_coeffs, rect, proj = calibrate_cams(cf['chessboard'],
                                                       cc,
                                                       fs)

    show_undistorted(cf['cameras'],
                     cam_mats,
                     dist_coeffs,
                     rect,
                     proj,
                     fs)