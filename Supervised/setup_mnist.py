## setup_mnist.py -- mnist data and model loading code
##


import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from keras import backend as K
from keras.initializers import Constant, glorot_normal
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Layer, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)+0.5
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)+0.5
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        self.train_data = train_data
        self.train_labels = train_labels


class MNISTModel:
    def __init__(self, restore, session=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        def fn(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct,logits=predicted)
       
        self.model = load_model(restore,custom_objects={'fn':fn})
        from keras.models import Model
        self.output = Model(input=self.model.input, outputs=self.model.layers[1].output)
        
    def predict(self, data):
        return self.model(data)
    def conv1(self, data):
        return self.output(data)
    
