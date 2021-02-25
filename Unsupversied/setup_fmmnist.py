#setup_fmmnist.py --load model and data
#author: Chia-Yi Hsu

import os
import numpy as np
import gzip
import urllib.request
import tensorflow as tf

from keras import backend as K
from keras.initializers import Constant, glorot_normal
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Layer, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

class ConcreteSelect(Layer):
    def __init__(self, output_dim=50, start_temp = 10, min_temp = 0.1, alpha = 0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.temp = self.add_weight(name = 'temp', shape = [], initializer = Constant(self.start_temp), trainable = False)
        self.logits = self.add_weight(name = 'logits', shape = [self.output_dim, input_shape[1]], initializer = glorot_normal(), trainable = True)
        super(ConcreteSelect, self).build(input_shape)
        
    def call(self, X, training = None):
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0)
        gumbel = -K.log(-K.log(uniform))
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        noisy_logits = (self.logits + gumbel) / temp
        samples = K.softmax(noisy_logits)
        
        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])
        
        self.selections = K.in_train_phase(samples, discrete_logits, training)
        Y = K.dot(X, K.transpose(self.selections))
        
        return Y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = {
        'output_dim': self.output_dim,
        'start_temp':self.start_temp,
        'min_temp':self.min_temp,
        'alpha': self.alpha}    
        base_config = super(ConcreteSelect, self).get_config()    
        return dict(list(base_config.items()) + list(config.items()))

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255.) 
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class FASHION_MNIST:
	def __init__(self):
		if not os.path.exists("data/fashion_mnist"):
			os.mkdir("data/fashion_mnist")
			files = ["train-images-idx3-ubyte.gz",
					"t10k-images-idx3-ubyte.gz",
					"train-labels-idx1-ubyte.gz",
					"t10k-labels-idx1-ubyte.gz"]
			for name in files:
				urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/' + name, "data/fashion_mnist/"+name)
		
		train_data = extract_data("data/fashion_mnist/train-images-idx3-ubyte.gz", 60000)
		train_labels = extract_labels("data/fashion_mnist/train-labels-idx1-ubyte.gz", 60000)
		self.test_data = extract_data("data/fashion_mnist/t10k-images-idx3-ubyte.gz", 10000)
		self.test_labels = extract_labels("data/fashion_mnist/t10k-labels-idx1-ubyte.gz", 10000)


		self.train_data = train_data
		self.train_labels = train_labels

class FMNISTModel:
    def __init__(self, restore, session=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
       
        self.model = load_model(restore,custom_objects={'ConcreteSelect':ConcreteSelect})
        from keras.models import Model
        self.output = Model(input=self.model.input, outputs=self.model.layers[1].output)
        
    def predict(self, data):
        data = tf.reshape(data,[-1,784])
        return tf.reshape(self.model(data),[-1,28,28,1])
		