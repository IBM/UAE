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

dir_model='models/mnist/convAE'

#Autoencoder
input_img = Input(shape=(28,28,1))    
    
encoded = Conv2D(16, (5,5), activation='relu', padding='same')(input_img)
encoded = MaxPooling2D((2,2), padding='same')(encoded)
encoded = Conv2D(8, (5,5), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D((2,2), padding='same')(encoded)

decoded = Conv2D(8, (5,5), activation='relu', padding='same')(encoded)
decoded = UpSampling2D((2,2))(decoded)
decoded = Conv2D(16, (5,5), activation='relu', padding='same')(decoded)
decoded = UpSampling2D((2,2))(decoded)
decoded = Conv2D(1, (5,5), activation='sigmoid', padding='same')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

data=MNIST()
x_train = data.train_data
x_test = data.test_data
history=autoencoder.fit(x_train, x_train, epochs=20, batch_size=100, 
                    shuffle=True)

autoencoder.save(dir_model)

