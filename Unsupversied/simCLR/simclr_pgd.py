#!/usr/bin/env python
# coding: utf-8
#author: Chia-Yi Hsu


import re
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import keras.backend as K
from setup_cifar import CIFAR


accessible tensor in the return dictionary

#load model
hub_path = 'original/hub/9766' #aug/hub/19532
module = hub.Module(hub_path, trainable=False)
inputs = tf.placeholder(tf.float32,[None,32,32,3])

#elements of pgd
logits =  module(inputs=inputs, signature="default", as_dict=True)['logits_sup'][:, :]
labels = tf.placeholder(tf.float32, [None, 10])
loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                       logits=logits)
grad = tf.gradients(loss, inputs)[0]


#load data
data = CIFAR()
idx = np.load('idx_std.npy')[:1000]
x_nat = data.test_data.astype('float32')[idx]
y = data.test_labels.astype('float32')[idx]

#pgd attack
step_size = 0.01
epsilons = [1/255., 2/255., 4/255., 6/255., 8/255., 10/255., 12/255.,14/255.,16/255.]
num_step=100
rand = True
adv=[]
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
sess.run(tf.global_variables_initializer())
acc=[]
for epsilon in epsilons:
    x_nat = data.test_data.astype('float32')[idx]
    if rand:
      x = x_nat + np.random.uniform(-epsilon, epsilon, x_nat.shape)
      x = np.clip(x, 0, 1) # ensure valid pixel range
    else:
      x = x_nat.astype(np.float)

    for i in range(num_step):
      grads, ls = sess.run([grad,loss],{inputs:x, labels:y})


      x = np.add(x, step_size * np.sign(grads), out=x, casting='unsafe')

      x = np.clip(x, x_nat - epsilon, x_nat + epsilon)
      x = np.clip(x, 0, 1)
    pred = sess.run(logits,{inputs:x , labels:y})
    pred = pred.argmax(-1)
    l = y.argmax(-1)
    acc.append(np.sum(pred==l))
    print(np.sum(pred==l))


# In[5]:


np.save('std_acc.npy',acc)


# In[ ]:




