import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('/home/jm/Code/haarcascades/haarcascade_frontalface_default.xml')


def grab_face(img):
    x = 20
    y = 20
    h = 100
    w = 100
    face_rectangle = face_cascade.detectMultiScale(img)

    #TODO: create a timing system to grab face in intervals / or at certain moments
    #TODO: create a wall panel of time elapse
    if len(face_rectangle) == 1:
        (x, y, h, w) = face_rectangle[0]
        return img[y:y + h, x:x + w].copy()
    else:
        return img[y:y + 20, x:x + 20].copy()


def print_face_coordinates(img):
    face_rectangles = face_cascade.detectMultiScale(img)
    for x, y, w, h in face_rectangles:
        print(x, y, w, h)


def detect_face(img):
    img_copy = img.copy()
    face_rects = face_cascade.detectMultiScale(img_copy)

    for (x, y, w, h) in face_rects:
        cv.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return img_copy
