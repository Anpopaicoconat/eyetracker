from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Concatenate
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import tensorflow
import matplotlib as plt
import numpy as np
import keras
import cv2
import os
#import img_processor as prcs
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

###############################################
batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 5 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout---------------------------------------------change
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons

###############################################

#model
eye_cbase = keras.models.load_model('open-close Sat Jan 18 18.27.05 2020.h5')
eye_cbase1 = keras.models.load_model('open-close Sat Jan 18 18.27.05 2020.h5')

#inputs
inp_og = Input(shape=(480, 640, 1)) #передаем чб предположительно важны только формы
inp_eye_l = Input(shape=(64, 64, 3)) # файнтюним ocm модель #будет ли он тренировать их раздельно?
inp_eye_r = Input(shape=(64, 64, 3)) # файнтюним ocm модель
inp_lm = Input(shape=(136,)) #под вопросом

#og conv base
og_conv_1 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(inp_og)
og_conv_2 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(og_conv_1)
og_pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(og_conv_2)
og_drop_1 = Dropout(drop_prob_1)(og_pool_1)

og_conv_3 = Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu')(og_drop_1)
og_conv_4 = Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu')(og_conv_3)
og_pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(og_conv_4)
og_drop_2 = Dropout(drop_prob_1)(og_pool_2)

og_flat = Flatten()(og_drop_2)
og_hidden = Dense(hidden_size, activation='relu')(og_flat)
og_drop_3 = Dropout(drop_prob_2)(og_hidden)

#land marks
lm_hidden = Dense(128, activation='relu')(inp_lm)
lm_drop_3 = Dropout(drop_prob_2)(lm_hidden)

#concatenate
lm_og_concatenate = Concatenate(axis=-1)([og_drop_3, lm_drop_3])
c1_hidden = Dense(128, activation='relu')(lm_og_concatenate)
c1_drop_3 = Dropout(drop_prob_2)(c1_hidden)

#eye conv base

ocm_out_l = eye_cbase.layers[-4].output
ocm_out_r = eye_cbase1.layers[-4].output

#concatenate
eye_concatenate = Concatenate(axis=-1)([ocm_out_l, ocm_out_r, c1_drop_3])
c2_hidden = Dense(hidden_size, activation='relu')(eye_concatenate)
c2_drop_3 = Dropout(drop_prob_2)(c2_hidden)

#out!
out = Dense(2, activation='softmax')(c2_drop_3)

#model
model = Model(input=[inp_og, inp_lm, inp_eye_l, inp_eye_r], output=out)

keras.utils.plot_model(model, 'main model.png')
#load data
def load():
    img_l, targets = prcs.load_images(r'C:\Users\anpopaicoconat\source\repos\detector\detector\data\coords\pasha 1')
    l_eye = []
    r_eye = []
    coord_l = []
    for i in img_l:
        l, r, c = prcs.search_eye(i)
        l_eye.append(l)
        r_eye.append(r)
        coord_l.append(c)
