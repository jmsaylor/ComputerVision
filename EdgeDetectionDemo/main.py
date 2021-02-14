import cv2
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread('/home/jm/Pictures/schwartz.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.blur(img, ksize=(17, 17))

    #Threshold
    _, thresh_0 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, thresh_1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh_2 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    _, thresh_3 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    _, thresh_4 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

    images = [img, thresh_0, thresh_1, thresh_2, thresh_3, thresh_4]

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 13))
    for ind, p in enumerate(images):
        ax = axs[ind//3, ind%3]
        ax.imshow(p)
    plt.show()

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
