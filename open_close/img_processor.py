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

def search_eye(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = detector(gray)
    face = detector(gray)[0]
    landmarks = predictor(gray, face)
    coords_l = []
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        coords_l.append(x)
        coords_l.append(y)
    eyes = ((landmarks.part(36).x, np.maximum(landmarks.part(37).y, landmarks.part(38).y), landmarks.part(39).x, np.minimum(landmarks.part(40).y, landmarks.part(41).y)),
            (landmarks.part(42).x, np.maximum(landmarks.part(43).y, landmarks.part(44).y), landmarks.part(45).x, np.minimum(landmarks.part(47).y, landmarks.part(46).y)))
    #(xr1 yr1 xr2 yr2) (xl1 yl1 xl2 yl2)
    ret = []
    for eye in eyes:
        # добиваем обрезку для одинакового разрешения
        dif_x = 64-(eye[2]-eye[0])#s otricatelnimi problema
        dif_y = 64-(eye[3]-eye[1])
        
        ad_xl = dif_x//2# нельзя увеличивать больше чем в 2 раза в этом случае произвести ресайз
        ad_xr = dif_x//2+dif_x%2
        
        ad_yl = dif_y//2
        ad_yr = dif_y//2+dif_y%2
        print('dif', dif_x, dif_y)
        eye_img = img[eye[1]-ad_yl: eye[3]+ad_yr, eye[0]-ad_xl: eye[2]+ad_xr]
        ret.append(eye_img)
    return ret[1], ret[0], coords_l # сначала левый потом правый относительно человека

def load_images(folder):
    images = []
    targets = []
    for filename in os.listdir(folder):
        print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        #img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_NEAREST)
        if img is not None:
            images.append(img)
            c = filename.split(')')[0].replace('(', '').replace(' ', '').split(',')
            targets.append(np.array(c))
    return images, targets

def show(img):
    while(True):
        k = cv2.waitKey()
        cv2.imshow("test", img)
        if k==27:    #выход esc
            break
        if k == -1:
            pass
        else:
            print(k)

img_l, _ = load_images(r'C:\Users\anpopaicoconat\source\repos\detector\detector\data\coords\pasha 1')
#eyes = []
#print('img_l', len(img_l))
def f():
    l_eye = []
    r_eye = []
    for i in img_l:
        l, r, c = search_eye(i)
        l_eye.append(l)
        r_eye.append(r)
        print(len(c))

    for img in r_eye:
        show(img)
