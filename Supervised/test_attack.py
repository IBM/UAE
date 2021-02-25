## test_attack.py -- sample code to perform UAE supervised attack (MinMax and Binary Search)
## author: Chiayi Hsu.

import tensorflow as tf
import numpy as np
import time,os,gc
os.environ["CUDA_VISIBLE_DEVICES"]='1'

from setup_mnist import MNIST, MNISTModel
 
from MINE_binary_search import MINE_supervised_binary_search
from supervised_attack import MINE_supervised





def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    np.random.seed(8)
    inputs = data.test_data
    np.random.shuffle(inputs)
    inputs = inputs[:samples]
    np.random.seed(8)
    targets = data.test_labels
    np.random.shuffle(targets)
    targets = targets[:samples]    

    return inputs, targets


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    #dataset
    data = MNIST()
    inputs, targets = generate_data(data, samples=1000, targeted=False,
                                        start=0, inception=False)

    #the path of classifier will be attacked
    dir_model='models/mnist'

    #the path of storing adversarial examples
    dir_adv = 'supervised_attack/result/'
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        
        model = MNISTModel(dir_model,sess)
        #MINE-based MinMax Algorithm
        attack = MINE_supervised(sess, model,batch_size=1, max_iterations=1000, confidence=0, 
            targeted=False, epsilon=1.0, mine_batch='conv') #mine_batch --'conv', 'random_sampling'

        #MINE-based Binary search Algorithm
        #attack = MINE_supervised_binary_search(sess, model, batch_size=1, max_iterations=1000, confidence=0,
        #    targeted=False, epsilon=1.0, mine_batch='conv')
        adv, Mi = attack.attack(inputs, targets)
        np.save(dir_adv+'/adv.npy',adv)
        np.save(dir_adv+'/mi.npy',Mi)
        
    tf.get_default_graph().finalize()
        