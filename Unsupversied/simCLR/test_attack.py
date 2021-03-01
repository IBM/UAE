## test_attack.py -- sample code to perform MINE-based unsupervised attack
##author: Chia-Yi Hsu

import tensorflow.compat.v1 as tf
import numpy as np
import time,os,gc
os.environ["CUDA_VISIBLE_DEVICES"]='2'

from setup_cifar import CIFAR
from setup_contrast import ContrastModel  

from contrast_attack import MINE_MinMax_unsupervised





def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    
    #idx = np.load('ID/MNIST/ID.npy')
    #idx = np.load('ID/mnist/9digit.npy')
    #inputs = data.test_data[idx][:samples]
    inputs = data.train_data
    #inputs=np.load('our attack/mnist/conv/conv1/training_data/AE/minmi_train.npy')
    #print(inputs.shape)
    #targets = data.test_labels[idx][:samples]
    targets = data.train_labels

    return inputs, targets


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    data = CIFAR()
    inputs, targets = generate_data(data, samples=1000, targeted=False,
                                        start=0, inception=False)
    mi=[]
    image=np.zeros([50000,32,32,3])
    idx=[]
    print('image: ',len(image))
    for i in range(image.shape[0]):
        if (image[i] ==0).all():
            print("start ",i)
            idx.append(i)
            #break
  
    start =0
    for i in idx:
        i = i+start
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
        
            model_dir = 'original/9766'
            model = ContrastModel(model_dir, session=None)
            
            
            print(i)
           
            attack = MINE_MinMax_unsupervised(sess, model, batch_size=1, max_iterations=20, confidence=0, 
                targeted=False, epsilon=1,aa=str(i))
            adv, Mi= attack.attack(inputs[i:i+315], targets[i:i+315])
         
            for j in range(adv.shape[0]):
                image[i-start+j] = adv[j].reshape(1,32,32,3)
            
            #save UAE
            np.save('our attack/cifar/optil2_train.npy',image)
            print('adv.shape:',adv.shape)
            
            del adv, attack, model
            #gc.collect()
        tf.get_default_graph().finalize()
   