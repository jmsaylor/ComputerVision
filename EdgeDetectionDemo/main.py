import cv2
import matplotlib.pyplot as plt
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # img = cv2.imread('/home/jm/Pictures/fibers.jpg')
    img = cv2.imread('/home/jm/Pictures/schwartz.jpg')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    img = cv2.blur(img, ksize=(17, 17))

    # Open/Close (Dilation/Erosion)


    # Morphological Transformations - Dilation

    kernel = np.ones((9, 9), np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=3)

    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(img_dilate, cmap='gray')
    plt.show()

    #Morphological Transformations - Erosion
    # kernel_0 = np.ones((9, 9), np.uint8)
    # kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # kernel_2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    #
    # kernels = [kernel_0, kernel_1, kernel_2]
    #
    # plt.figure(figsize=(20, 20))
    # for i in range(3):
    #     img_copy = img.copy()
    #     img_copy = cv2.erode(img_copy, kernels[i], iterations=3)
    #     plt.subplot(1, 3, i+1)
    #     plt.imshow(img_copy)
    #     plt.axis('off')
    # plt.show()

    #Gradients Example
    # sobel_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=5)
    # sobel_y =cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=5)
    # blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y, beta=0.5, gamma=0)
    # laplacian = cv2.Laplacian(img, cv2.CV_64F)
    #
    # images = [sobel_x, sobel_y, blended, laplacian]
    # plt.figure(figsize= (20,20))
    # for i in range(4):
    #     plt.subplot(1, 4, i+1)
    #     plt.imshow(images[i], cmap='gray')
    #     plt.axis('off')
    # plt.show()


    #Threshold Example
    # _, thresh_0 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # _, thresh_1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # _, thresh_2 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    # _, thresh_3 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    # _, thresh_4 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    #
    # images = [img, thresh_0, thresh_1, thresh_2, thresh_3, thresh_4]
    #
    # fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 13))
    # for ind, p in enumerate(images):
    #     ax = axs[ind//3, ind%3]
    #     ax.imshow(p)
    # plt.show()

    #Blur 2
    # img_0 = cv2.blur(img, ksize=(7, 7))
    # img_1 = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=0)
    # img_2 = cv2.medianBlur(img, 7)
    # img_3 = cv2.bilateralFilter(img, 7, sigmaSpace=75, sigmaColor=75)
    #
    # images = [img_0, img_1, img_2, img_3]
    #
    # figs, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 20))
    #
    # for ind, p in enumerate(images):
    #     ax = axs[ind]
    #     ax.imshow(p)
    #     ax.axis('off')
    # plt.show()

    # Blur 1
    # kernels = [5, 11, 17]
    #
    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
    #
    # for ind, s in enumerate(kernels):
    #     img_blurred = cv2.blur(img, ksize=(s, s))
    #     ax = axs[ind]
    #     ax.imshow(img_blurred)
    #     ax.axis('off')
    # plt.show()
