import json

with open('config.json', 'r') as f:
    config = json.load(f)


def config_build():
    config = {}

    # WINDOW SETTINGS
    config['windows'] = {'ORIGIN_X': 1440,
                         'ORIGIN_Y': 0}

    # ROI SETTINGS
    config['roi'] = {'ADJUST_SCALE': 2,
                     'PADDING': 1.5}

    # find_circle SETTINGS
    config['find_circle'] = {'BLUR_K': 1,
                             'HOUGH_DP': 1.2,
                             'HOUGH_PARAM1': 150,
                             'HOUGH_PARAM2': 1}

    # print message SETTINGS
    config['messages'] = {
        'key_input': 'Press SPACE to accept and proceed to next step, '
                     'ESC to abort step and go to previous step, and \'q\' '
                     'to abort and end calibration procedure',
        'capture_image': 'Hold up calibration target(s) and press SPACE to '
                         'capture image.',
        'drag_circle': 'Drag a circle from the center of desired target to '
                       'the edge and release mouse.  Circle can be '
                       'finely adjusted in next step, so it does not have to '
                       'be perfect.',
        'adjust_circle': 'Adjust the circle so that it is coradial with the '
                         'tracking target.',
        'adjust_hsv_values': 'Adjust the HSV threshold limits until the '
                             'target is highly visible and the rest of the '
                             'image is mostly masked.  Press SPACE to '
                             'confirm, ESC to abort.'
    }

    # hsv percentile threshold SETTINGS
    config['thresh_percs'] = [[5, 95],  # (H_LO, H_HI)
                              [5, 95],  # (S_LO, S_HI)
                              [5, 95]]  # (V_LO, V_HI)

    # initial 'k' value for image blurring
    config['blur_k'] = {'initial': 10,
                        'max': 20}


    with open('config.json', 'w') as file:
        json.dump(config, file)


if __name__ == "__main__":
    config_build()
