"""DOCSTRING"""
from config import config as cf
import cv2
from imutils.video import WebcamVideoStream
import numpy as np
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
        self.size = get_image_size(self.id)
        self.cam_mat = None
        self.dist_coeff = None

    def calibrate(self, chessboard, frame_count):

        image_points = chessboard_cap(self, chessboard, frame_count)

        object_pts = chessboard.expand_points(image_points)

        self.cam_mat, self.dist_coeff = calibrate_intrinsic(object_pts,
                                                            image_points,
                                                            self.size)

        show_undistorted(self)

        return


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

    def __init__(self, cams, chessboard, frame_count):
        self.cams = cams
        self.chessboard = chessboard
        self.frame_count = frame_count

        self.rot_mat = None
        self.trans_vec = None
        self.rect_mat = None
        self.proj_mat = None

        self.corner_list = None


# def calibrate_cameras(chessboard, corner_list, frame_sizes):
#     """DOCSTRING"""
#
#     real_pts = [chessboard_points(chessboard)] * len(corner_list[0])
#
#     cam_mats = []
#     dist_coeffs = []
#
#     for corners, frame_size in zip(corner_list, frame_sizes):
#         __, cam_mat, dist_coeff, __, __ = \
#             cv2.calibrateCamera(real_pts,
#                                 corners,
#                                 tuple(frame_size), 0, 0)
#
#         cam_mats.append(cam_mat)
#         dist_coeffs.append(dist_coeff)
#
#     return cam_mats, dist_coeffs


# def calibrate_cams(chessboard, corner_list, frame_sizes):
#     """Calibrate cameras using chessboard corner data.  Determines intrinsic
#     and extrinsic parameters of stereo camera configuration.
#     """
#
#     rot_mat, trans_vec = calibrate_stereo(chessboard,
#                                           corner_list,
#                                           frame_sizes)[:2]
#
#     print(rot_mat)
#     print(trans_vec)
#
#     # rect = [None] * 2
#     # proj = [None] * 2
#
#     rect = [None] * 2
#     proj = [None] * 2
#
#     [rect[0], rect[1], proj[0], proj[1]] = \
#         cv2.stereoRectify(cam_mats[0],
#                           dist_coeffs[0],
#                           cam_mats[1],
#                           dist_coeffs[1],
#                           tuple(frame_sizes[0]),
#                           rot_mat,
#                           trans_vec)[:4]
#
#     print(rect[0], '\n', rect[1])
#     print(proj[0], '\n', proj[1])
#
#     return cam_mats, dist_coeffs, rect, proj


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


def capture_chessboard_images(cam_ids, chessboard, num_frames):
    """Capture frames from input cameras of chessboard calibration rig."""

    stream_prefix = 'Camera'
    cap_prefix = 'Capture'

    streams = initialize_streams(cam_ids)
    stream_windows = initialize_windows(stream_prefix, cam_ids)
    cap_windows = initialize_windows(cap_prefix, cam_ids)

    focus_window(stream_windows[0])

    frames = []

    while len(frames) < num_frames:

        capture_instruct(len(frames), num_frames)
        images, keypress = capture_image(streams, stream_windows)
        retvals, corners = find_corners(images, cap_windows,
                                        chessboard.dims)

        # clean this up???
        if keypress == 32:  # SPACE
            if all(retvals):
                frames.append(images)
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

    return frames


def capture_image(streams, windows):
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
    if current_frame < total_frames:
        print('Press SPACE to capture.')
    else:
        print('Press SPACE to finish capturing frames.')
    print('Press ESC to go back one frame.')
    print('Press \'q\' to quit.')

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

    streams = initialize_streams(cam_list)
    stream_windows = initialize_windows(stream_prefix, cam_list)
    cap_windows = initialize_windows(cap_prefix, cam_list)

    focus_window(stream_windows[0])

    frames = []
    corner_list = []

    while len(frames) <= num_frames:

        capture_instruct(len(frames), num_frames)
        images, keypress = capture_image(streams, stream_windows)
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

    kill_streams(streams)

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


# keep as Chessboard class method??  Remove if so
def get_chessboard_points(spacing, dims):
    """ Builds matrix of 3D points with locations of chessboard corners.
    Z-coordinates are zero.
    """

    # initialize (H*V) x 3 matrix of points
    points = np.zeros((dims[0] * dims[1], 3), np.float32)

    # populate x,y coordinates with 2D corner locations and reshape
    points[:, :2] = spacing * (np.mgrid[0:dims[0],
                               0:dims[1]].T.reshape(-1, 2))

    return points


def get_image_size(cam_id):
    """Get image size of input camera ID"""

    cap = cv2.VideoCapture(cam_id)

    __, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()

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


def initialize_cameras(cam_ids):
    """DOCSTRING"""

    cameras = []

    for cam_id in cam_ids:
        cameras.append(Camera(cam_id))

    return cameras


def initialize_streams(cameras):
    """Initialize webcam streams on input cameras.  Uses multithreading to
    prevent camera desync due to camera buffers."""

    streams = []
    for cam in cameras:
        streams.append(WebcamVideoStream(cam.id).start())

    return streams


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

    streams = initialize_streams(cameras)
    stream_windows = initialize_windows(stream_prefix, cameras)

    focus_window(stream_windows[0])

    keypress = -1

    while keypress == -1:

        for i, (stream, m1, m2) in enumerate(zip(streams, map1, map2)):
            cap = stream.read()
            cap = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
            cv2.imshow(stream_windows[i], cap)

            cv2.remap(cap, m1, m2, cv2.INTER_LINEAR, udists[i])
            # udists[i] = cv2.remap(cap, m1, m2, cv2.INTER_LINEAR)

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

    streams = initialize_streams(cameras)
    stream_windows = initialize_windows(stream_prefix, cameras)
    dst_windows = initialize_windows('dst', cameras)

    focus_window(stream_windows[0])

    keypress = -1

    while keypress == -1:

        for stream, swin, dwin, m1, m2 in zip(streams,
                                              stream_windows,
                                              dst_windows,
                                              map1,
                                              map2):
            cap = stream.read()
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

    streams = initialize_streams(cam_list)
    stream_windows = initialize_windows(stream_prefix, cam_list)

    focus_window(stream_windows[0])

    keypress = -1

    while keypress == -1:

        for stream, win, camera in zip(streams, stream_windows, cam_list):
            image = stream.read()

            dst = cv2.undistort(image, camera.cam_mat, camera.dist_coeff)

            cv2.imshow(win, dst)

        keypress = cv2.waitKey(5)

    kill_streams(streams)
    cv2.destroyAllWindows()

    return


def to_list(element):
    """If input is not a list, converts to single element list."""

    if type(element) is not list:
        return [element]
    else:
        return element


# TESTING FUNCTIONS - delete later
def test_calib_intrinsic_save():

    cameras = initialize_cameras(cf['cam_ids'])

    chessboard = Chessboard(cf['chessboard']['spacing'],
                            cf['chessboard']['dims'])

    for cam in cameras:
        cam.calibrate(chessboard, cf['stereo_cal_frames'])

    intrinsic_data = {'cameras': cameras}

    with open('intrinsic_data', 'wb') as fp:
        pickle.dump(intrinsic_data, fp)

    return


def test_cornercap_save(cameras):
    """testing function"""

    chessboard = Chessboard(cf['chessboard']['spacing'],
                            cf['chessboard']['dims'])

    stereo = Stereo(cameras, chessboard, cf['stereo_cal_frames'])

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
    plt.plot(corner_list[0][0][:, :, 0].T[0], corner_list[0][0][:, :, 1].T[0], 'bo')
    plt.plot(corner_list[1][0][:, :, 0].T[0], corner_list[1][0][:, :, 1].T[0], 'go')
    plt.show()

    return


if __name__ == "__main__":

    calib_intrinsic = False

    if calib_intrinsic:
        test_calib_intrinsic_save()

    cameras = test_open_saved('intrinsic_data', 'cameras')

    for camera in cameras:
        print(camera.cam_mat)
        print(camera.dist_coeff)

    cap_frames = True

    if cap_frames:
        test_cornercap_save(cameras)

    cameras, chessboard, corner_list = test_open_corner_data()

    # test_compare_pts(chessboard, corner_list)

    r, t, e, f = calibrate_stereo(cameras, chessboard, corner_list)

    print('r \n', r)
    print('t \n', t)

    #  testing reassignment
    # r = np.eye(3)
    # t = np.asarray([-120.0, 0.0, 0.0]).T

    # show_undistorted(cameras)
    #
    rect, proj = get_rect_matrices(cameras, r, t)

    print('rect \n', rect[0], '\n', rect[1])
    print('proj \n', proj[0], '\n', proj[1])

    map1, map2 = get_rect_maps(cameras, rect, proj)

    show_rectified(cameras, rect, proj, map1, map2)

    print(cameras[0].size)
    print(map1[0].size)
    print(map2[0].size)

    show_remap(cameras, map1, map2)