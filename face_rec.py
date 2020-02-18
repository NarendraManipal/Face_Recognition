import cv2
import os
import numpy as np


def faceDetection(test_img):
    #convertion to gray image
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    #haar classifier detect object
    face_haar_casecade = cv2.CasecadeClassifier('harr_cascade/haarcascade_frontalface_default.xml')
    faces = face_haar_casecade.detectMultiScale(gray_img, scaleFactor = 1.32, minNeighbors = 5)

    return faces, gray_img
    
