import cv2

drawing = False
ix = -1
iy = -1

def draw_circle(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, center=(x, y), radius=5, color=(0,0,255), thickness=-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, center=(x, y), radius=10, color=(255, 0, 0), thickness=1)

def draw_rectangle(event, x, y, flags, param):

    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, pt1=(ix, iy), pt2=(x,y), color=(255, 200, 180), thickness=-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, pt1=(ix, iy), pt2=(x, y), color=(255, 200, 180), thickness=-1)

if __name__ == '__main__':

    #Dynamic IO Example
    img = cv2.imread('/home/jm/Pictures/world.jpg')

    cv2.namedWindow(winname='my_drawing')
    cv2.setMouseCallback('my_drawing', draw_circle)

    while True:
        cv2.imshow('my_drawing', img)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# Color Conversions
# img = cv2.imread('/home/jm/Pictures/fibers.jpg')
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# plt.imshow(img_gray, cmap='gray')

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
# ax1.imshow(img_hsv)
# ax2.imshow(img_hls)


# Drawing Shapes Example
# loveWall = cv2.imread('/home/jm/Pictures/lovewall.jpg', cv2.COLOR_BGR2RGB)
#
# cv2.circle(loveWall, center=(100, 180), radius=40, color=(0, 255, 0), thickness=2)
# cv2.rectangle(loveWall, pt1=(350, 350), pt2=(420, 380), color=(255, 255, 0), thickness=3)
#
# plt.imshow(loveWall)




















