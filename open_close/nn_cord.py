
import matplotlib as plt
import numpy as np
import keras
import cv2
import os

#model
inp_og = Input(shape=(1, 480, 640)) #передаем чб предположительно важны только формы
inp_eye = Input(shape=(3, 64, 64))
inp_lm = Input(shape=(68, 2)) #под вопросом

