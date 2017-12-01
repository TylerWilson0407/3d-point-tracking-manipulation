from config import config
import gui_functions
import cv2
import numpy as np

# for testing
# import matplotlib.pyplot as plt

def test_frame():
    im_num = 1
    leftright = 'L'
    imfile = r'test_images/ballcalib_' + str(im_num) + '_' + leftright + '.bmp'

    return cv2.imread(imfile)


def circle_mouse_callback(event, x, y, flags, params):
    """Mouse callback function for selecting a circle on a frame by dragging
    from the center to the edge"""
    if event == cv2.EVENT_LBUTTONDOWN and not params['drag']:
        params['point1'] = np.array([x, y])
        params['drag'] = True
    elif event == cv2.EVENT_MOUSEMOVE and params['drag']:
        params['point2'] = np.array([x, y])
    elif event == cv2.EVENT_LBUTTONUP and params['drag']:
        params['point2'] = np.array([x, y])
        params['drag'] = False
        if not np.array_equal(params['point2'], params['point1']):
            params['selected'] = True
    return


def drag_circle(img):
    """Select a region of interest from a captured frame"""
    wd_frame = 'Select Calibration Targets'
    cv2.namedWindow(wd_frame)
    cv2.moveWindow(wd_frame,
                   config['windows']['ORIGIN_X'],
                   config['windows']['ORIGIN_Y'])

    center = None
    radius = None

    cb_params = {'drag': None,
                 'point1': np.zeros(2),
                 'point2': np.zeros(2),
                 'selected': None}

    while not cb_params['selected']:
        cv2.imshow(wd_frame, img)

        cv2.setMouseCallback(wd_frame, circle_mouse_callback,
                             param=cb_params)

        if cb_params['drag']:
            circ_img = img.copy()
            radius = np.int(np.linalg.norm(cb_params['point1'] -
                                           cb_params['point2']))
            cv2.circle(circ_img,
                       tuple(cb_params['point1']),
                       radius,
                       (0, 0, 0), 1, 8, 0)
            cv2.imshow(wd_frame, circ_img)

        center = cb_params['point1']
        radius = np.int(np.linalg.norm(cb_params['point1'] -
                                       cb_params['point2']))

        if cv2.waitKey(5) == 27:
            break

    circle = [center,
              radius]
    return circle


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


def adjust_circle(image, roi, roi_origin, initial_circle):
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

    roi = image[(center[1] - rad_padded):
                (center[1] + rad_padded),
                (center[0] - rad_padded):
                (center[0] + rad_padded)]

    roi_origin = np.array([(center[1] - rad_padded),
                           (center[0] - rad_padded)])

    return roi, roi_origin


def select_circle(image):
    """Specify a circle area by first dragging a rough circle then finely
    adjusting the circle.  Returns (EXPAND HERE)
    """
    confirmed = False
    circle_adj = None

    while not confirmed:
        circle = drag_circle(image)
        roi, roi_origin = get_circle_roi(image, circle)
        circle_adj = adjust_circle(image, roi, roi_origin, circle)
        confirmed = gui_functions.button_confirm()

    return circle_adj


def circle_mask(image, circle):
    """Create a binary mask where all points outside the input circle are
    set to zero(black).
    """
    mask = np.zeros_like(image)
    cv2.circle(mask, circle[0], circle[1], (1, 1, 1), -1)
    masked = image * mask
    return masked


image = test_frame()
test_circle = select_circle(image)
print(test_circle)
mask_circ = circle_mask(image, test_circle)
# mask_circ = image
cv2.imshow('masked circle', mask_circ)
cv2.waitKey(0)

image_hsv = cv2.cvtColor(mask_circ, cv2.COLOR_BGR2HSV)
print(image_hsv)

###
roi_hsv = cv2.cvtColor(mask_circ,cv2.COLOR_BGR2HSV)
roi_h = cv2.split(roi_hsv)[0]
roi_s = cv2.split(roi_hsv)[1]
roi_v = cv2.split(roi_hsv)[2]

iHeight,iWidth,iDepth = roi_hsv.shape
# h = np.zeros((iHeight,iWidth,iDepth))
h = np.zeros([256, 256,3])
bins = np.arange(256).reshape(256,1)
color = [ (255,0,0),(0,255,0),(0,0,255)]

test_hist = np.zeros([3, 256, 1])

for ch,col in enumerate(color):
    hist_item = cv2.calcHist([roi_hsv],[ch],None,[256],[0,255])
    hist_item[0, :] = 0
    hist_item = np.int32(hist_item)
    hist_item_norm = cv2.normalize(hist_item,0,255,
                                 cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item_norm))
    pts = np.column_stack((bins,hist))
    cv2.polylines(h,[pts],False,col)
    test_hist[ch] = hist_item

h = np.flipud(h)

cv2.imshow('crop',mask_circ)
cv2.imshow('roi hue',roi_h)
cv2.imshow('roi sat',roi_s)
cv2.imshow('roi val',roi_v)
cv2.imshow('colorhist',h)
cv2.moveWindow('crop', image.shape[1] + 18 * 1 + 200, 200)
cv2.moveWindow('roi hue', image.shape[1] + mask_circ.shape[1] + 18 * 2 +
               200, 200)
cv2.moveWindow('roi sat', image.shape[1] + 18 * 1 + 200, mask_circ.shape[0]
               + 200)
cv2.moveWindow('roi val', image.shape[1] + mask_circ.shape[1] + 18 * 2 + 200,
               mask_circ.shape[0] + 200)
cv2.moveWindow('colorhist', image.shape[1] + 2 * mask_circ.shape[1] + 18 * 3
               + 200,
               200)
if h.shape[0] < 256 and h.shape[1] < 256:
    h = h[0:255, 0:255]
    cv2.resizeWindow('colorhist', 256, 256)
cv2.waitKey(0)

perc_range = range(101)
perc_vals = np.zeros(101)
for i in perc_range:
    perc_vals[i] = np.percentile(roi_h, i)

def get_percentile_bin(hist, pct):
    cs = np.cumsum(hist)
    print(cs)
    bin_id = np.searchsorted(cs, np.percentile(cs, pct))
    return bin_id

test_bin1 = get_percentile_bin(test_hist[0], 10)
test_bin2 = get_percentile_bin(test_hist[0], 95)
print(test_hist[0])
print(test_bin1, test_bin2)
print(test_hist[0][test_bin1], test_hist[0][test_bin2])


###