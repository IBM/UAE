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

dir_model = 'models/mnist/sparseAE'

input_img = Input(shape=(784,))
encoded = Dense(128,activation='sigmoid',activity_regularizer=regularizers.l1(10e-8))(input_img)
decoded = Dense(784, activation='sigmoid',activity_regularizer=regularizers.l1(10e-8))(encoded)

autoencoder = Model(input_img, decoded)

def kl_divergence(p, p_hat):
    return p*K.log(p) - p*K.log(p_hat) + (1-p)*K.log(1-p) - (1-p)*K.log(1-p_hat)
def sparse_loss(layer1_output):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1) + 3*K.mean(kl_divergence(0.1, K.mean(K.clip(layer1_output,1e-10,1.),axis=0)))
    return loss

autoencoder.compile(optimizer='adam',loss=sparse_loss(encoded))

data=MNIST()
shape=784
x_train = data.train_data.reshape(-1,shape)
x_test = data.test_data.reshape(-1,shape)

history=autoencoder.fit(x_train, x_train, epochs=50, batch_size=100, 
                    shuffle=True)

autoencoder.save(dir_model)
