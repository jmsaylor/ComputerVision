import cv2 as cv
import detect_hand

if __name__ == '__main__':

    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read(0)

        img = detect_hand.detectHand(frame)

        cv.imshow('Hand Detection', img)

        if 27 == cv.waitKey(1):
            break

    cap.release()
    cv.destroyAllWindows()