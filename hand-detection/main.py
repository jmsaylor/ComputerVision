import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread('/home/jm/Pictures/hand.jpeg')




    # Contour Detection
    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,48,80], dtype='uint8')
    upper = np.array([20, 255, 255], dtype='uint8')
    skinRegionHSV = cv2.inRange(hsv, lower, upper)

    #Thresholds
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    #Contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    # cv2.drawContours(img, [contours], -1, (0,255,0), 2)

    # Convex Hull

    hull = cv2.convexHull(contours)
    cv2.drawContours(img, [hull], -1, (0,255,0), 2)


    plt.imshow(img, cmap='gray')
    plt.show()

