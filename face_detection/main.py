import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('/home/jm/Code/haarcascades/haarcascade_frontalface_default.xml')

def detect_face(img):
	img_copy = img.copy()
	face_rects = face_cascade.detectMultiScale(img_copy)

	for (x, y, w, h) in face_rects:
		cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0,255,0), 3)

	return img_copy

cap = cv2.VideoCapture(0)

while True:

	ret, frame = cap.read(0)

	frame = detect_face(frame)
	cv2.imshow('Video Face Detection', frame)

	c = cv2.waitKey(1)

	if c == 27:
		break

cap.release()
cv2.destroyAllWindows()
