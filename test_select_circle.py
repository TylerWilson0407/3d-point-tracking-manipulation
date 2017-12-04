from config import config
import cv2
import numpy as np


class TrackingTarget:
    def __init__(self, camera):
        self.c = cv2.VideoCapture(camera)
        self._cal_count = None
        self._cal_funcs = None
        self._cal_args = None
        self._cal_init = None
        self._cal_image = None

        self._circle = None

        self._roi = None
        self._roi_origin = None

        self._adj_circle = None

        self.cb_params = {'drag': None,
                          'point1': np.zeros(2),
                          'point2': np.zeros(2),
                          'selected': None}

        self._circle = None

        self.thresh_percs = config['thresh_percs']

        self.thresholds = None

    def calibrate(self):
        """Calibrate the tracking target.  Captures an image from the video
        feed and allows the user to drag a circular region of interest where
        the target is and finely adjust it.  Once the circular ROI is
        selected, an upper and lower percentile of the HSV values of the
        region of interested are calculated, which are used in threshold
        masks to find the target and subsequently track it."""
        self._cal_count = 0

        """List of functions.  ***explain why using a list with cal_count to be
        able to jump back and forth in the calibration steps, as well as the 
        cal_init list"""
        self._cal_funcs = [self.capture_image,
                           self.drag_circle,
                           self.adjust_circle,
                           self.adjust_hsv_thresholds]

        self._cal_args = [[],
                          [],
                          [],
                          []]

        self._cal_init = [False] * len(self._cal_funcs)

        while 0 <= self._cal_count < len(self._cal_funcs):
            self._cal_funcs[self._cal_count](*self._cal_args[self._cal_count])

    def capture_image(self):
        """Displays video from input camera feed"""
        # window initialization
        win_name = 'Camera Feed'
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name,
                       config['windows']['ORIGIN_X'],
                       config['windows']['ORIGIN_Y'])

        print_instruct(config['messages']['capture_image'])

        # initialize variables
        keypress = -1
        frame = None

        # Display video feed unless Space or Esc are pressed
        while keypress == -1:
            __, frame = self.c.read()
            cv2.imshow(win_name, frame)
            keypress = cv2.waitKey(10)

        if keypress == 32:  # 32 = Space
            print('Image captured.')
            self._cal_image = frame
            self._cal_count += 1
        elif keypress == 27:  # 27 = Esc
            print('Calibration aborted.')
            self._cal_count -= 1

        cv2.destroyWindow(win_name)
        return

    def drag_circle(self):
        """Select a region of interest from a captured frame"""
        # window initialization
        win_name = 'Select Calibration Targets'
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name,
                       config['windows']['ORIGIN_X'],
                       config['windows']['ORIGIN_Y'])

        print_instruct(config['messages']['drag_circle'])

        # initialize variables
        center = None
        radius = None
        self.cb_params['selected'] = None

        # run loop until circle is dragged and released
        while not self.cb_params['selected']:
            cv2.imshow(win_name, self._cal_image)

            # mouse callback function for dragging circle on window
            cv2.setMouseCallback(win_name, circle_mouse_callback,
                                 param=self)

            # continuously draw circle on image while mouse is being dragged
            if self.cb_params['drag']:
                circ_img = self._cal_image.copy()
                radius = np.int(np.linalg.norm(self.cb_params['point1'] -
                                               self.cb_params['point2']))
                cv2.circle(circ_img,
                           tuple(self.cb_params['point1']),
                           radius,
                           (0, 0, 0), 1, 8, 0)
                cv2.imshow(win_name, circ_img)

            center = self.cb_params['point1']
            radius = np.int(np.linalg.norm(self.cb_params['point1'] -
                                           self.cb_params['point2']))

            # abort function if Esc pressed
            if cv2.waitKey(5) == 27:
                print('Drag Circle aborted.')
                self._cal_count -= 1
                break
        else:
            print('Drag Circle complete.')
            self._circle = [center,
                            radius]
            self._cal_count += 1

        cv2.destroyWindow(win_name)
        return

    def adjust_circle(self):
        """Manually adjust a circle on an image"""

        print_instruct(config['messages']['adjust_circle'])

        roi, roi_origin = get_circle_roi(self._cal_image, self._circle)

        circle_local = np.copy(self._circle)
        circle_local[0] = self._circle[0] - np.flipud(roi_origin)

        scale = config['roi']['ADJUST_SCALE']
        roi = scale_image(roi, scale)
        circle_local = np.multiply(circle_local, scale)

        wd_adjcir = 'Adjust Target Circle'
        keypress = -1
        img_circ = np.copy(roi)
        # Set max radius of circle as half of the longest side of image
        max_radius = np.max([roi.shape[0], roi.shape[1]]) // 2
        cv2.namedWindow(wd_adjcir)
        cv2.resizeWindow(wd_adjcir, 200, 200)
        cv2.createTrackbar('x', wd_adjcir,
                           circle_local[0][0], roi.shape[1], empty_callback)
        cv2.createTrackbar('y', wd_adjcir,
                           circle_local[0][1], roi.shape[0], empty_callback)
        cv2.createTrackbar('r', wd_adjcir,
                           circle_local[1], max_radius, empty_callback)
        while keypress == -1:
            cv2.circle(img_circ,
                       (cv2.getTrackbarPos('x', wd_adjcir),
                        cv2.getTrackbarPos('y', wd_adjcir)),
                       cv2.getTrackbarPos('r', wd_adjcir),
                       (0, 0, 0),
                       1)
            cv2.imshow(wd_adjcir, img_circ)
            img_circ = np.copy(roi)
            keypress = cv2.waitKey(5)

        if keypress == 32:
            print('Circle adjusted.')
            self._adj_circle = ((cv2.getTrackbarPos('x', wd_adjcir) // scale +
                                 roi_origin[1],
                                 cv2.getTrackbarPos('y', wd_adjcir) // scale +
                                 roi_origin[0]),
                                cv2.getTrackbarPos('r', wd_adjcir) // scale)
            self._cal_count += 1
        elif keypress == 27:  # 27 = Esc
            print('Adjust Circle aborted.')
            self._cal_count -= 1

        cv2.destroyWindow(wd_adjcir)
        return

    def adjust_hsv_thresholds(self):
        """Displays thresholded HSV binary and allows user to adjust
        threshold limits."""

        print_instruct(config['messages']['adjust_hsv_values'])

        win_track = 'Adjustment Control Panel'
        cv2.namedWindow(win_track)

        masked_circle = circle_mask(self._cal_image, self._adj_circle)
        self.thresholds = get_hsv_thresholds(masked_circle, self.thresh_percs)

        thresh_names = ['H_LO', 'H_HI',
                        'S_LO', 'S_HI',
                        'V_LO', 'V_HI']

        blur_k_name = 'BLUR \'K\''

        max_vals = [179, 179, 255, 255, 255, 255]

        for thresh, val, max_val in zip(thresh_names,
                                        self.thresholds,
                                        max_vals):
            cv2.createTrackbar(thresh, win_track, val, max_val, empty_callback)

        cv2.createTrackbar(blur_k_name, win_track,
                           config['blur_k']['initial'],
                           config['blur_k']['max'],
                           blur_nonzero_callback)

        keypress = -1

        while keypress == -1:
            __, frame = self.c.read()

            # blur frame
            blur_k = cv2.getTrackbarPos(blur_k_name, win_track)
            # set blur_k to 1 if less than 1, since trackbar lower limit is 0.
            if blur_k < 1:
                blur_k = 1
            frame_blur = cv2.blur(frame, (blur_k, blur_k))

            frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

            thresh_vals = np.asarray([cv2.getTrackbarPos(name, win_track)
                                      for name in thresh_names])

            frame_thresh = cv2.inRange(frame_hsv,
                                       thresh_vals[::2],
                                       thresh_vals[1::2])

            # cv2.imshow(win_thresh, frame_thresh)
            cv2.imshow(win_track, frame_thresh)

            keypress = cv2.waitKey(5)

        if keypress == 27:
            print('Adjust HSV Values aborted.')
            self._cal_count -= 1
        elif keypress == 32:
            print('HSV values saved.')
            self._cal_count += 1
        else:
            print('Values reset to default.')

        cv2.destroyWindow(win_track)


def print_instruct(message):
    print('*' * 79)
    print(message)
    print(config['messages']['key_input'])
    print('*' * 79)


def blur_nonzero_callback(x):
    """Return 1 if x < 1, otherwise return x.  Used as trackbar callback for
    values that cannot go to zero(e.g. blur 'k' value."""
    if x < 1:
        return 1
    else:
        return x


def test_frame():
    im_num = 1
    leftright = 'L'
    imfile = r'test_images/ballcalib_' + str(im_num) + '_' + leftright + '.bmp'

    return cv2.imread(imfile)


# noinspection PyUnusedLocal
def circle_mouse_callback(event, x, y, flags, params):
    """Mouse callback function for selecting a circle on a frame by dragging
    from the center to the edge"""
    if event == cv2.EVENT_LBUTTONDOWN and not params.cb_params['drag']:
        params.cb_params['point1'] = np.array([x, y])
        params.cb_params['drag'] = True
    elif event == cv2.EVENT_MOUSEMOVE and params.cb_params['drag']:
        params.cb_params['point2'] = np.array([x, y])
    elif event == cv2.EVENT_LBUTTONUP and params.cb_params['drag']:
        params.cb_params['point2'] = np.array([x, y])
        params.cb_params['drag'] = False
        if not np.array_equal(params.cb_params['point2'],
                              params.cb_params['point1']):
            params.cb_params['selected'] = True
    return


def scale_image(image, scale):
    """Resize an image by a given scale."""
    x_scale = int(np.around(image.shape[1] * scale))
    y_scale = int(np.around(image.shape[0] * scale))
    scaled_image = cv2.resize(image, (x_scale, y_scale))
    return scaled_image


def empty_callback(*_, **__):
    """Empty callback function for passing to functions that require a
    callback.  Accepts any argument and returns None.
    """
    return None


def adjust_circle(roi, roi_origin, initial_circle):
    """Manually adjust a circle on an image"""

    circle_local = initial_circle
    circle_local[0][0] = initial_circle[0][0] - roi_origin[1]
    circle_local[0][1] = initial_circle[0][1] - roi_origin[0]

    scale = config['roi']['ADJUST_SCALE']
    roi = scale_image(roi, scale)
    circle_local = np.multiply(circle_local, scale)

    wd_adjcir = 'Adjust Target Circle'
    keypress = -1
    img_circ = np.copy(roi)
    # Set max radius of circle as half of the longest side of image
    max_radius = np.max([roi.shape[0], roi.shape[1]]) // 2
    cv2.namedWindow(wd_adjcir)
    cv2.resizeWindow(wd_adjcir, 200, 200)
    cv2.createTrackbar('x', wd_adjcir,
                       circle_local[0][0], roi.shape[1], empty_callback)
    cv2.createTrackbar('y', wd_adjcir,
                       circle_local[0][1], roi.shape[0], empty_callback)
    cv2.createTrackbar('r', wd_adjcir,
                       circle_local[1], max_radius, empty_callback)
    while keypress == -1:
        cv2.circle(img_circ,
                   (cv2.getTrackbarPos('x', wd_adjcir),
                    cv2.getTrackbarPos('y', wd_adjcir)),
                   cv2.getTrackbarPos('r', wd_adjcir),
                   (0, 0, 0),
                   1)
        cv2.imshow(wd_adjcir, img_circ)
        img_circ = np.copy(roi)
        keypress = cv2.waitKey(5)

    if keypress == 27:
        return None

    adj_circle = ((cv2.getTrackbarPos('x', wd_adjcir) // scale +
                   roi_origin[1],
                   cv2.getTrackbarPos('y', wd_adjcir) // scale +
                   roi_origin[0]),
                  cv2.getTrackbarPos('r', wd_adjcir) // scale)

    cv2.destroyWindow(wd_adjcir)
    return adj_circle


def get_circle_roi(image, circle):
    """Get a region of interest around a defined circle in an image."""
    center = circle[0]
    radius = circle[1]
    padding = config['roi']['PADDING']
    rad_padded = int(radius * padding)

    y_min = max((center[1] - rad_padded), 0)
    y_max = min((center[1] + rad_padded), image.shape[0])
    x_min = max((center[0] - rad_padded), 0)
    x_max = min((center[0] + rad_padded), image.shape[1])

    roi = image[y_min: y_max, x_min: x_max]

    roi_origin = np.array([y_min, x_min])

    return roi, roi_origin


def get_hsv_thresholds(image, percentiles):
    """Find an upper and lower threshold for each HSV channel given an
    image and upper and lower percentiles."""

    if len(percentiles) == 1:
        percentiles = [percentiles] * image.shape[2]
    elif len(percentiles) != image.shape[2]:
        raise IndexError('Number of percentile ranges must be either 1 or '
                         'match number of channels in image.')

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_thresholds = []
    for channel, perc in zip(cv2.split(image_hsv), percentiles):
        hsv_thresholds.append(channel_percentile(channel, perc))

    # flatten list
    flat_thresh = [item for sublist in hsv_thresholds for item in sublist]

    return flat_thresh


def circle_mask(image, circle):
    """Create a binary mask where all points outside the input circle are
    set to zero(black).
    """
    center = circle[0]
    radius = circle[1]

    padding = config['roi']['PADDING']
    pad_radius = int(padding * radius)

    mask = np.zeros_like(image)
    cv2.circle(mask, center, radius, (1, 1, 1), -1)
    # reversed because circle center is indexed (x, y) where image is (y, x)
    roi_min = reversed(np.subtract(center, pad_radius))
    roi_max = reversed(np.add(center, pad_radius))
    frame_min = [0, 0]
    frame_max = image.shape
    mins = [max(*v) for v in zip(roi_min, frame_min)]
    maxs = [min(*v) for v in zip(roi_max, frame_max)]
    masked = image * mask
    return masked[mins[0]: maxs[0], mins[1]: maxs[1]]


def channel_percentile(channel, percentile, rem_zero=True):
    """Returns a percentile value for a single-channel image.  Optional
    parameter rem_zero will remove all zero values before calculating the
    percentile."""
    flat_vals = np.ndarray.flatten(channel)
    sorted_vals = np.sort(flat_vals)

    if rem_zero:
        values = np.trim_zeros(sorted_vals)
    else:
        values = sorted_vals

    perc_vals = []
    for perc in percentile:
        perc_vals.append(np.int(np.percentile(values, perc)))

    return perc_vals


# def get_percentile_bin(hist, pct):
#     cs = np.cumsum(hist)
#     bin_id = np.searchsorted(cs, np.percentile(cs, pct))
#     return bin_id

# image = test_frame()
# test_circle = select_circle(image)
# print(test_circle)
# mask_circ = circle_mask(image, test_circle)
#
# cv2.imshow('masked circle', mask_circ)
# cv2.waitKey(0)
#
# np.save('test_roi_save', mask_circ)
# np.save('test_image_save', image)
#
#
# image_hsv = cv2.cvtColor(mask_circ, cv2.COLOR_BGR2HSV)
#
# roi_hsv = cv2.cvtColor(mask_circ,cv2.COLOR_BGR2HSV)
# roi_h = cv2.split(roi_hsv)[0]
# roi_s = cv2.split(roi_hsv)[1]
# roi_v = cv2.split(roi_hsv)[2]
#
#
# ch_perc = channel_percentile(roi_h, [5, 95])
#
# h = np.zeros([256, 256,3])
# bins = np.arange(256).reshape(256,1)
# color = [ (255,0,0),(0,255,0),(0,0,255)]
#
# test_hist = np.zeros([3, 256, 1])
#
# for ch,col in enumerate(color):
#     hist_item = cv2.calcHist([roi_hsv],[ch],None,[256],[0,255])
#     hist_item[0, :] = 0
#     hist_item = np.int32(hist_item)
#     hist_item_norm = cv2.normalize(hist_item,0,255,
#                                  cv2.NORM_MINMAX)
#     hist=np.int32(np.around(hist_item_norm))
#     pts = np.column_stack((bins,hist))
#     cv2.polylines(h,[pts],False,col)
#     test_hist[ch] = hist_item
#
# h = np.flipud(h)
#
# cv2.imshow('crop',mask_circ)
# cv2.imshow('roi hue',roi_h)
# cv2.imshow('roi sat',roi_s)
# cv2.imshow('roi val',roi_v)
# cv2.imshow('colorhist',h)
# cv2.moveWindow('crop', image.shape[1] + 18 * 1 + 200, 200)
# cv2.moveWindow('roi hue', image.shape[1] + mask_circ.shape[1] + 18 * 2 +
#                200, 200)
# cv2.moveWindow('roi sat', image.shape[1] + 18 * 1 + 200, mask_circ.shape[0]
#                + 200)
# cv2.moveWindow('roi val', image.shape[1] + mask_circ.shape[1] + 18 * 2 + 200,
#                mask_circ.shape[0] + 200)
# cv2.moveWindow('colorhist', image.shape[1] + 2 * mask_circ.shape[1] + 18 * 3
#                + 200,
#                200)
# if h.shape[0] < 256 and h.shape[1] < 256:
#     h = h[0:255, 0:255]
#     cv2.resizeWindow('colorhist', 256, 256)
# cv2.waitKey(0)
#
# perc_range = range(101)
# perc_vals = np.zeros(101)
# for i in perc_range:
#     perc_vals[i] = np.percentile(roi_h, i)
#
# a = test_hist[0]
# nz_hist = np.all(a == 0)
# print(nz_hist)
#
# print(ch_perc)
# print(a[ch_perc[0]], a[ch_perc[1]])
#
# ###

target = TrackingTarget(0)
target.calibrate()
