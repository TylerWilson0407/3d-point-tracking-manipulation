import cv2
import numpy as np

im_num = 1
leftright = 'L'
imfile = r'test_images\ballcalib_' + str(im_num) + '_' + leftright + '.bmp'

img = cv2.imread(imfile)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur_k = 1
img_blur = cv2.blur(img_gray, (blur_k, blur_k))

dp = 10
min_dist = 50
param1 = 200
param2 = 125
min_rad = 15
max_rad = 35
# circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist)
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                           param1=param1, param2=param2,
                           minRadius=min_rad, maxRadius=max_rad)
print(circles)

img_circ = np.copy(img)
for circ in circles[0]:
    print(circ)
    cv2.circle(img_circ, (circ[0], circ[1]), circ[2], 5, 2)

# cv2.circle(frame1, pt1, 5, 255, -1)

origin = [20, 20]
hor_space = 20
ver_space = 50

def window_pos(index, init, spacing, frame_shape):
    pos = init + index * (frame_shape + spacing)
    return pos

cv2.namedWindow('Raw')
cv2.namedWindow('Grayscale')
cv2.namedWindow('Blur')
cv2.namedWindow('Circles')

cv2.moveWindow('Raw',
               window_pos(0, origin[0], hor_space, img.shape[1]),
               window_pos(0, origin[1], ver_space, img.shape[0]))
cv2.moveWindow('Grayscale',
               window_pos(1, origin[0], hor_space, img.shape[1]),
               window_pos(0, origin[1], ver_space, img.shape[0]))
cv2.moveWindow('Blur',
               window_pos(0, origin[0], hor_space, img.shape[1]),
               window_pos(1, origin[1], ver_space, img.shape[0]))
cv2.moveWindow('Circles',
               window_pos(1, origin[0], hor_space, img.shape[1]),
               window_pos(1, origin[1], ver_space, img.shape[0]))

cv2.imshow('Raw', img)
cv2.imshow('Grayscale', img_gray)
cv2.imshow('Blur', img_blur)
cv2.imshow('Circles', img_circ)

while cv2.waitKey(5) != 27:
    pass
