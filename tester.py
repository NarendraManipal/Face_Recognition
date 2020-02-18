import cv2
import numpy as np
import face_recognition as fr
import os


test_img = cv2.imread('img/Narendra.jpg', 0)
faces, gray_img = fr.faceDetection(test_img)
print("Face Detected: ", faces)

for (x, y, w, h) in faces:
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness = 5)

resized_image = cv2.resize(test_img, (1000, 700))
cv2.imshow("Face Detection: ", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows
