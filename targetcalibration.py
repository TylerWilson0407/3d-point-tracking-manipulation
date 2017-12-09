from config import config as cf
# Remove this later if merging TrackingTarget into calibration module
from calibrate_track_target import TrackingTarget
import cv2
import numpy as np


class TargetCalibration:
    def __init__(self):
        self.cams = create_cam_dict(cf['cameras'])

        # names for targets - short for index, middle, thumb(fingers)
        targ_names = ['index', 'middle', 'thumb']
        self.targs = create_tracking_targets(targ_names, self.cams)

    def calibrate(self):
        """EXPAND HERE when you figure out how to layout the entire
        calbration procedure(e.g. break up into camera and target
        calibration functions?).
        """


def create_cam_dict(cam_ids):
    """Output dictionary with indexed keys(cam0, cam1, etc) with values of
    the camera IDs.
    """
    cams = {}
    for i, cam in enumerate(cam_ids):
        key = 'cam' + str(i)
        cams[key] = cam
    return cams


def create_tracking_targets(keys, cams):
    """Output dictionary with *keys as keys and TrackingTarget objects as
    values."""

    targs = {}
    for key in keys:
        targs[key] = {}
        for cam in cams:
            targs[key][cam] = TrackingTarget()

    print(targs)
    return targs


# TESTING
if __name__ == "__main__":
    cal = TargetCalibration()
