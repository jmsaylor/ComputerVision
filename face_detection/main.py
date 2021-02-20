import cv2
import numpy as np

from slice import grab_face, print_face_coordinates, detect_face
cap = cv2.VideoCapture(0)
# i = 0
while True:
	# i += 1
	ret, frame = cap.read(0)

	# face = detect_face(frame)
	face = grab_face(frame)
	cv2.imshow('Video Face Detection', face)
	# print_face_coordinates(frame)

	c = cv2.waitKey(1)

	if c == 27:
		break

cap.release()
cv2.destroyAllWindows()
