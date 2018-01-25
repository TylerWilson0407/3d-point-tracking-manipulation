import json

with open('config.json', 'r') as f:
    config = json.load(f)


def config_build():
    conf = {

        # Chessboard parameters
        'chessboard': {'dist': 23.3,  # mm between corners
                       'hor_sq': 8,  # horizontal # of squares
                       'ver_sq': 6},

        # Number of frames to capture for stereo camera calibration
        'stereo_calib': {'num_frames': 20},

        # Device IDs for cameras
        'cameras': [0, 1],

        # Multiprocessing SETTINGS
        # set number of processes for multiprocessing, 0 will use number of
        # cores on machine
        'multiprocessing': {'processes': 0},

        # three tracking targets
        # RULES: Don't have any target names starting with 'q' and have each
        # target name start with a different letter
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
