import cv2 as cv
import numpy as np

def detectHand(img):
    img_copy = img.copy()
    img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype='uint8')
    upper = np.array([20, 255, 255], dtype='uint8')
    skinRegionHSV = cv.inRange(img_copy, lower, upper)

    blurred = cv.blur(skinRegionHSV, (4,4))
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))

    height, width, channels = img.shape

    blank_image = np.zeros((height, width, channels), np.uint8)

    contours = cv.convexHull(contours)

    cv.drawContours(blank_image, [contours], -1, (0,255,0), 3)

    return blank_image
