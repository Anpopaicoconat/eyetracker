
from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import cv2
import os
import random
import time
import matplotlib.pyplot as plt


def show(i):
    cv2.imshow('{}'.format(data_y[i]), data_x[i])
    cv2.waitKey()
    print(data_y[i])

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        #img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_NEAREST)
        #img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_NEAREST)
        if img is not None:
            images.append(img)
    return images

def mk_model():
    inp = Input(shape=(depth, height, width)) # N.B. depth goes first in Keras!
    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    conv_1 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(inp)
    conv_2 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    conv_3 = Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    drop_2 = Dropout(drop_prob_1)(pool_2)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(num_classes, activation='softmax')(drop_3)

    model = Model(input=inp, output=out) # To define a model, just specify its input and output layers

    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam', # using the Adam optimiser
                  metrics=['accuracy']) # reporting the accuracy
    return model

SEED = random.randint(0, 1000)
data_o = load_images_from_folder('C:/Users/anpopaicoconat/source/repos/detector/detector/data/open/processed')
data_c = load_images_from_folder('C:/Users/anpopaicoconat/source/repos/detector/detector/data/close/processed')
data_y = [0 for i in range(len(data_o))]+[1 for i in range(len(data_c))]
random.seed(SEED)
data_x = data_o + data_c
random.shuffle(data_x)
random.seed(SEED)
random.shuffle(data_y)#0 -> open, 1 -> closed





###############################################
k = 4
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
num_val_samples = len(data_x) // k
data_x = np.array(data_x).astype('float32')

X_train = np.array(data_x)
y_train = np.array(data_y)
X_test = np.array(data_x)
y_test = np.array(data_y)

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

data_x /= 255 # Normalise data to [0, 1] range


###############################################
score=[]
for i in range(k):
    X_test = data_x[i*num_val_samples:(i+1)*num_val_samples]
    Y_test = data_y[i*num_val_samples:(i+1)*num_val_samples]

    print('shape', np.array(data_x[:i*num_val_samples]).shape, np.array(data_x[:(i+1)*num_val_samples]).shape)
    X_train = np.concatenate([data_x[:i*num_val_samples],
                             data_x[:(i+1)*num_val_samples]],
                             axis=0)
    Y_train = np.concatenate([data_y[:i*num_val_samples],
                             data_y[:(i+1)*num_val_samples]],
                             axis=0)
    num_train, depth, height, width = X_train.shape 
    num_test = X_test.shape[0] 
    num_classes = np.unique(y_train).shape[0] 
    Y_train = np_utils.to_categorical(Y_train, num_classes) # One-hot encode the labels
    Y_test = np_utils.to_categorical(Y_test, num_classes) # One-hot encode the labels
    print(Y_train)
    model = mk_model()
    history = model.fit(X_train, Y_train, 
              batch_size=batch_size, epochs=num_epochs) 
    val_mse, val_mae = model.evaluate(X_test, Y_test, verbose=1) 
    score.append(val_mae)
model.save('open-close {}.h5'.format(time.ctime(time.time()).replace(':', '.')))
#print('saved')
print('score', score, np.mean(score))

history_dict = history.history
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf() #Очистить рисунок
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()