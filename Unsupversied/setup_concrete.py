##setup_concrete.py ----load model and data
##author: Chia-Yi Hsu

from keras.utils import to_categorical
from keras import backend as K
from keras.initializers import Constant, glorot_normal
from keras.layers import Dense, Dropout, Activation, Flatten, Layer, Input
from keras.models import load_model

import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

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

class ISOLET:
	def __init__(self):
		x_train = np.genfromtxt('data/isolet/isolet1+2+3+4.data', delimiter = ',', usecols = range(0, 617), encoding = 'UTF-8')
		y_train = np.genfromtxt('data/isolet/isolet1+2+3+4.data', delimiter = ',', usecols = [617], encoding = 'UTF-8')
		x_test = np.genfromtxt('data/isolet/isolet5.data', delimiter = ',', usecols = range(0, 617), encoding = 'UTF-8')
		y_test = np.genfromtxt('data/isolet/isolet5.data', delimiter = ',', usecols = [617], encoding = 'UTF-8')

		X = MinMaxScaler(feature_range=(0,1)).fit_transform(np.concatenate((x_train, x_test)))
		x_train = X[: len(y_train)]
		x_test = X[len(y_train):]
      
		self.train_data = x_train
		self.train_labels = to_categorical(y_train)
		self.test_data = x_test
		self.test_labels = to_categorical(y_test)
class ISOLETModel:
	"""docstring for ClassName"""
	def __init__(self, restore, session=None):
		self.num_channels = 1
		self.image_size = 617
		self.num_labels = 27

		self.model = load_model(restore, custom_objects={'ConcreteSelect':ConcreteSelect})

	def predict(self,data):
		return self.model(data)

		
class COIL:
	def __init__(self):
		samples = []
		for i in range(1, 21):
			for image_index in range(72):
				obj_img = Image.open(os.path.join('data/coil-20-proc', 'obj%d__%d.png' % (i, image_index)))
				rescaled = obj_img.resize((20,20))
				pixels_values = [float(x) for x in list(rescaled.getdata())]
				sample = np.array(pixels_values + [i])
				samples.append(sample)
		samples = np.array(samples)
		np.random.shuffle(samples)
		data = samples[:, :-1]
		targets = (samples[:, -1] + 0.5).astype(np.int64)
		data = (data - data.min()) / (data.max() - data.min())

		l = data.shape[0] * 4 // 5
		train = (data[:l], to_categorical(targets[:l]))
		test = (data[l:], to_categorical(targets[l:]))
      
		(self.train_data, self.train_labels) = train
		(self.test_data, self.test_labels) = test

class COILModel:
	"""docstring for ClassName"""
	def __init__(self, restore, session=None):
		self.num_channels = 1
		self.image_size = 400
		self.num_labels = 21

		self.model = load_model(restore, custom_objects={'ConcreteSelect':ConcreteSelect})

	def predict(self, data):
		return self.model(data)


class MICE:
	def __init__(self,one_hot=False):
		filling_value = -100000

		X = np.genfromtxt('data/Data_Cortex_Nuclear.csv', delimiter = ',', skip_header = 1, usecols = range(1, 78), filling_values = filling_value, encoding = 'UTF-8')
		classes = np.genfromtxt('data/Data_Cortex_Nuclear.csv', delimiter = ',', skip_header = 1, usecols = range(78, 81), dtype = None, encoding = 'UTF-8')

		for i, row in enumerate(X):
			for j, val in enumerate(row):
				if val == filling_value:
					X[i, j] = np.mean([X[k, j] for k in range(classes.shape[0]) if np.all(classes[i] == classes[k])])

		DY = np.zeros((classes.shape[0]), dtype = np.uint8)
		for i, row in enumerate(classes):
			for j, (val, label) in enumerate(zip(row, ['Control', 'Memantine', 'C/S'])):
				DY[i] += (2 ** j) * (val == label)

		Y = np.zeros((DY.shape[0], np.unique(DY).shape[0]))
		for idx, val in enumerate(DY):
			Y[idx, val] = 1

		X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)

		indices = np.arange(X.shape[0])
		np.random.shuffle(indices)
		X = X[indices]
		Y = Y[indices]
		DY = DY[indices]
		classes = classes[indices]

		if not one_hot:
			Y = DY

		X = X.astype(np.float32)
		Y = Y.astype(np.float32)
      
		self.train_data = X[: X.shape[0] * 4 // 5]
		self.train_labels = to_categorical(Y[: X.shape[0] * 4 // 5])
		self.test_data = X[X.shape[0] * 4 // 5:]
		self.test_labels = to_categorical(Y[X.shape[0] * 4 // 5: ])

class MICEModel:
	"""docstring for ClassName"""
	def __init__(self, restore, session=None):
		self.num_channels = 1
		self.image_size = 77
		self.num_labels = 8

		self.model = load_model(restore, custom_objects={'ConcreteSelect':ConcreteSelect(output_dim=10)})

	def predict(self, data):
		return self.model(data)

class ACTIVITY:
	def __init__(self):
		x_train = np.loadtxt(os.path.join('data/dataset_uci', 'final_X_train.txt'), delimiter = ',', encoding = 'UTF-8')
		x_test = np.loadtxt(os.path.join('data/dataset_uci', 'final_X_test.txt'), delimiter = ',', encoding = 'UTF-8')
		y_train = np.loadtxt(os.path.join('data/dataset_uci', 'final_y_train.txt'), delimiter = ',', encoding = 'UTF-8')
		y_test = np.loadtxt(os.path.join('data/dataset_uci', 'final_y_test.txt'), delimiter = ',', encoding = 'UTF-8')

		X = MinMaxScaler(feature_range=(0,1)).fit_transform(np.concatenate((x_train, x_test)))
		x_train = X[: len(y_train)]
		x_test = X[len(y_train):]

		self.train_data = x_train
		self.train_labels = to_categorical(y_train)
		self.test_data = x_test
		self.test_labels = to_categorical(y_test)

class ACTIVITYModel:
	"""docstring for ClassName"""
	def __init__(self, restore, session=None):
		self.num_channels = 1
		self.image_size = 561
		self.num_labels = 7

		self.model = load_model(restore, custom_objects={'ConcreteSelect':ConcreteSelect})

	def predict(self, data):
		return self.model(data)
