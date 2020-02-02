
import matplotlib as plt
import numpy as np
import keras
import cv2
import os
import img_processor as prcs

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


ocm = keras.models.load_model('open-close Sat Jan 18 18.27.05 2020.h5')

img_l = prcs.load_images(r'C:\Users\anpopaicoconat\source\repos\detector\detector\data\coords\pasha 2')
eyes = []
Y_test = []
print('img_l', len(img_l))
for i in img_l[0]:
    
    for j in prcs.search_eye(i):
        eyes.append(cv2.resize(j, (32, 32), interpolation = cv2.INTER_NEAREST))
        Y_test.append([0, 1])
        

