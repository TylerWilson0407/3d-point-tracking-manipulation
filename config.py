import json

with open('config.json', 'r') as f:
    config = json.load(f)

def config_build():
    config = {}

    # WINDOW SETTINGS
    config['windows'] = {'ORIGIN_X': 20,
                         'ORIGIN_Y': 200}

    #ROI SETTINGS
    config['roi'] = {'ADJUST_SCALE': 2,
                     'PADDING': 1.5}

    # find_circle SETTINGS
    config['find_circle'] = {'BLUR_K': 1,
                             'HOUGH_DP': 1.2,
                             'HOUGH_PARAM1': 150,
                             'HOUGH_PARAM2': 1}

    with open('config.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    config_build()
