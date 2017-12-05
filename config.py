import json

with open('config.json', 'r') as f:
    config = json.load(f)


def config_build():
    config = {}

    # Device IDs for cameras
    config['cameras'] = [0, 2]

    # three tracking targets
    config['targets'] = ['pos',  # positioning target
                         'ori',  # orienting target
                         'sta']  # stabilizing target

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
