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




def create_tracking_targets(*keys):
    """Output dictionary with *keys as keys and TrackingTarget objects as
    values."""

    targs = {}
    for key in keys:
        targs[key] = TrackingTarget

    return targs
