import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Conv2DTranspose,Reshape, Layer
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from setup_mnist  import MNIST
from setup_cifar import CIFAR
from setup_svhn import SVHN
import keras.backend as K
from keras import regularizers

import sys
import numpy as np
import tensorflow as tf

dir_model = 'models/mnist/denseAE'

#autoencoder
input_img = Input(shape=(784,))
encoded = Dense(128,activation='sigmoid')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

from keras.optimizers import RMSprop
opt =RMSprop(lr=0.0025)
autoencoder.compile(optimizer=opt, loss='mse')

data=MNIST()
shape=784
x_train = data.train_data.reshape(-1,shape)
x_test = data.test_data.reshape(-1,shape)

history=autoencoder.fit(x_train, x_train, epochs=20, batch_size=100, 
                    shuffle=True)
autoencoder.save(dir_model)