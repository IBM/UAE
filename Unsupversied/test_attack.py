##test_attack.py ---- sample code to perform MINE-based unsupervised attack
##author: Chia-Yi Hsu

import tensorflow as tf
import numpy as np
import time,os,gc
os.environ["CUDA_VISIBLE_DEVICES"]='1'


from setup_mnist import MNIST, MNISTModel
from setup_svhn import SVHN, SVHNModel
from setup_fmmnist import FASHION_MNIST, FMNISTModel
from setup_concrete import COIL, COILModel, MICE, MICEModel, ACTIVITY, ACTIVITYModel  

from unsupervised_attack import MINE_unsupervised




if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    #the path of storing UAE
    dir_adv='unsupervised_attack/'

    #load model of Autoencoder
    dir_model = 'models/MNIST/convAE'
    
    data = MNIST()
    inputs = data.train_data
    

    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        
        model = MNISTModel(dir_model,sess)
        attack = MINE_unsupervised(sess, model,batch_size=1, max_iterations=40, confidence=0, 
            epsilon=1.0, mine_batch='conv')
        adv, Mi = attack.attack(inputs)
        
        np.save(dir_adv+'adv.npy',image)
        np.save(dir_adv+'mi.npy',Mi)
        
    tf.get_default_graph().finalize()
     