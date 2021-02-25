from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU, Softmax,Layer
from keras.initializers import Constant, glorot_normal
from keras.models import load_model, Model
from keras import backend as K
import numpy as np
import os
from setup_mnist import MNIST
from setup_fmmnist import FASHION_MNIST
os.environ["CUDA_VISIBLE_DEVICES"]='0'

dir_model='models/mnist/concreteAE'

def f(x):
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(784)(x)
    return x

data=MNIST()
shape=784
x_train = data.train_data.reshape(-1,shape)
x_test = data.test_data.reshape(-1,shape)

selector = ConcreteAutoencoderFeatureSelector(K = 50, output_function = f, num_epochs = 800)
selector.fit(x_train, x_train, x_test, x_test)

selector.model.save(dir_model)