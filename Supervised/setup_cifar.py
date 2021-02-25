## setup_cifar.py -- cifar data and model loading code
##



import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request


def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255.))
    return np.array(images),np.array(labels)
    

class CIFAR:
    def __init__(self):
        train_data = []
        train_labels = []
        
        if not os.path.exists("cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()
            

        for i in range(5):
            r,s = load_batch("cifar-10-batches-bin/data_batch_"+str(i+1)+".bin")
            train_data.extend(r)
            train_labels.extend(s)
            
        train_data = np.array(train_data,dtype=np.float32)
        train_labels = np.array(train_labels)
        
        self.test_data, self.test_labels = load_batch("cifar-10-batches-bin/test_batch.bin")
        
        self.train_data = train_data
        self.train_labels = train_labels

class CIFARModel:
    def __init__(self, restore, session=None):
        self.num_channels = 3
        self.image_size = 32
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

        
    
