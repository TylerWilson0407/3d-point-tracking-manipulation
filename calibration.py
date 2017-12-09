from config import config
# Remove this later if merging TrackingTarget into calibration module
from calibrate_track_target import TrackingTarget
import cv2
import numpy as np


class Calibration:
    def __init__(self):
        self.cams = config['cameras']

        # names for targets - short for positioning, orienting, stabilizing
        targ_names = ['pos', 'ori', 'sta']
        self.targs = create_tracking_targets(*targ_names)


    def calibrate(self):
        """EXPAND HERE when you figure out how to layout the entire
        calbration procedure(e.g. break up into camera and target
        calibration functions?)."""

        # threshlist = dict((k, v) for k, v in self.targs.iteritems() if
        #                   v.thresholds)


        # if not all()


def create_tracking_targets(*keys):
    """Output dictionary with *keys as keys and TrackingTarget objects as
    values."""

    targs = {}
    for key in keys:
        targs[key] = TrackingTarget()

    return targs
