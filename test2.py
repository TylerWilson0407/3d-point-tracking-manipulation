import cv2
from imutils.video import WebcamVideoStream
import time


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


def get_image_size(cam_id):
    """Get image size of input camera ID"""

    cap = cv2.VideoCapture(cam_id)

    __, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()

    return frame.shape


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


def kill_streams(streams):
    """Kill stream threads."""

    for stream in streams:
        stream.stop()
        while not stream.stream.isOpened():
            return
    return


if __name__ == "__main__":
    cameras = initialize_cameras([0, 1])
    streams = initialize_streams(cameras)
    streams = initialize_streams(cameras)
    kill_streams(streams)
    # streams = initialize_streams(cameras)