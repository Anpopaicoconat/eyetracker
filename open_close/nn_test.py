
import matplotlib as plt
import numpy as np
import keras
import cv2
import os

def prediction(img):
    img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_NEAREST)
    img = np.array([img])
    img = img.astype('float32') 
    img /= np.max(img)
    p = ocm.predict(img)
    if p[0][0]<p[0][1]:
        p='                                                                           закрыты'
    else:
        p='открыты'
    return p


ocm = keras.models.load_model('open-close.h5')

camera = 'http://192.168.43.229:4747/=http://192.168.43.229:4747/video'
cap = cv2.VideoCapture('http://192.168.43.229:4747/video')
