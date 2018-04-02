import json

with open('config.json', 'r') as f:
    config = json.load(f)


def config_build():
    conf = {

        'calib_data_filename': 'calibration_data',

        # Chessboard parameters
        # Note that chessboard dims needs to be tuple but tuples are not
        # preserved with JSON.  Need to convert to tuple after
        'chessboard': {'spacing': 25,  # mm between corners
                       'dims': [8, 6]},  # (horiz squares, vert squares)

        # Number of frames to capture for stereo camera calibration
        'calib_frame_count': {'intrinsic': 3,
                              'stereo': 1},

        # Device IDs for cameras
        'cam_ids': [0, 1],

        # Multiprocessing SETTINGS
        # set number of processes for multiprocessing, 0 will use number of
        # cores on machine
        'multiprocessing': {'processes': 0},

        # three tracking targets
        # RULES: Don't have any target names starting with 'q' and have each
        # target name start with a different letter(this is due to how
        # keypress inputs are used in the calibration process)
        'targets': ['index',
                    'middle',
                    'thumb'],

        # WINDOW SETTINGS
        'windows': {'ORIGIN_X': 1440, 'ORIGIN_Y': 0},

        # ROI SETTINGS
        'roi': {'ADJUST_SCALE': 2,
                'PADDING': 1.5},

        # find_circle SETTINGS
        'find_circle': {'BLUR_K': 1,
                        'HOUGH_DP': 1.2,
                        'HOUGH_PARAM1': 150,
                        'HOUGH_PARAM2': 1},

        # hsv percentile threshold SETTINGS
        'thresh_percs': [[5, 95],  # (H_LO, H_HI)
                         [5, 95],  # (S_LO, S_HI)
                         [5, 95]],

        # initial 'k' value for image blurring
        'blur_k': {'initial': 10,
                   'max': 20}

    }

    with open('config.json', 'w') as file:
        json.dump(conf, file)


if __name__ == "__main__":
    config_build()