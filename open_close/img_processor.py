#это модуль препроцессинга изображений
import numpy as np
import cv2
import os
import dlib
import time
import keras

#загрузка моделей
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def search_f(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        #поиск глаз
        landmarks = predictor(gray, face)
        eyes = ((landmarks.part(36).x, np.maximum(landmarks.part(37).y, landmarks.part(38).y), landmarks.part(39).x, np.minimum(landmarks.part(40).y, landmarks.part(41).y)),
               (landmarks.part(42).x, np.maximum(landmarks.part(43).y, landmarks.part(44).y), landmarks.part(45).x, np.minimum(landmarks.part(47).y, landmarks.part(46).y)))
        xr1 = landmarks.part(36).x
        yr1 = np.maximum(landmarks.part(37).y, landmarks.part(38).y)
        xr2 = landmarks.part(39).x
        yr2 = np.minimum(landmarks.part(40).y, landmarks.part(41).y)

        xl1 = landmarks.part(42).x
        yl1 = np.maximum(landmarks.part(43).y, landmarks.part(44).y)
        xl2 = landmarks.part(45).x
        yl2 = np.minimum(landmarks.part(47).y, landmarks.part(46).y)

        # добиваем обрезку для одинакового разрешения
        dif_x = 64-(x2-x1)#s otricatelnimi problema
        dif_y = 64-(y2-y1)
        
        ad_xl = dif_x//2
        ad_xr = dif_x//2+dif_x%2
        
        ad_yl = dif_y//2
        ad_yr = dif_y//2+dif_y%2