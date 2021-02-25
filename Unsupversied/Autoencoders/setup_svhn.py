##setup_svhn.py-------load data and model
##author: Chia-Yi Hsu

import numpy as np
import os
import scipy.io as sio
import urllib.request
import keras
from keras.models import load_model
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten, Layer, Input
import tensorflow as tf

class SVHN:
    def __init__(self):
        if not os.path.exists("data/SVHN"):
            os.mkdir("data/SVHN")
            files = ["train_32x32.mat",
            		 "test_32x32.mat",
            		 "extra_32x32.mat"]
            for name in files:

                urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/'+ name, "data/SVHN/"+name)


        train_data = np.transpose(sio.loadmat("data/SVHN/train_32x32.mat")["X"], (3,0,1,2)) / 255.
        train_labels = sio.loadmat("data/SVHN/train_32x32.mat")["y"]
        train_labels[train_labels == 10] = 0
        train_labels = keras.utils.to_categorical(train_labels,10)
        
        self.test_data = np.transpose(sio.loadmat("data/SVHN/test_32x32.mat")["X"], (3,0,1,2)) / 255.
        self.test_labels = sio.loadmat("data/SVHN/test_32x32.mat")["y"]
        self.test_labels[self.test_labels == 10] = 0
        self.test_labels = keras.utils.to_categorical(self.test_labels,10)
           
        self.train_data = train_data
        self.train_labels = train_labels

class SVHNModel:
    def __init__(self, restore, session=None):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        def kl_divergence(p, p_hat):
            return p*K.log(p) - p*K.log(p_hat) + (1-p)*K.log(1-p) - (1-p)*K.log(1-p_hat)
        def sparse_loss(layer1_output):
            def loss(y_true, y_pred):
                return K.mean(K.square(y_pred - y_true), axis=-1) + K.mean(kl_divergence(0.1, K.mean(K.clip(layer1_output,1e-10,1.),axis=0)))
            return loss
        #loss of sparse AE
        #self.model = load_model(restore, custom_objects={'loss':sparse_loss(Input(shape=(256,)))})

        #without customizing loss
        self.model = load_model(restore)
        from keras.models import Model
        self.output = Model(input=self.model.input, outputs=self.model.layers[1].output)
        
    def predict(self, data):
    	#return tf.reshape(self.model(tf.reshape(data,[-1,3072])), [-1,32,32,3])
        return self.model(data)
    def conv1(self, data):
        return self.output(data)