## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = False       # if we stop improving, abort gradient descent early
LEARNING_RATE = 0.01     # larger values converge faster to less accurate results
TARGETED = False          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess
def standardized(img):
    mean, var = tf.nn.moments(tf.convert_to_tensor(img),[1])
    mean = tf.tile(tf.reshape(mean,[-1,1]),[1,tf.shape(img)[1]])
    var = tf.tile(tf.reshape(tf.sqrt(var),[-1,1]),[1, tf.shape(img)[1]])
    img = (img-mean)/var
    return img
def MiNetwork(x_in, y_in,mine_batch_='conv'):
    H=10
    seed = np.random.randint(0,1000,1)
    
    y_shuffle = tf.gather(y_in, tf.random_shuffle(tf.range(tf.shape(y_in)[0]),seed=seed))
    x_conc = tf.concat([x_in, x_in], axis=0)
    y_conc = tf.concat([y_in, y_shuffle], axis=0)
    
    # propagate the forward pass
    if mine_batch_ == 'conv':
        layerx = tf.layers.conv2d(x_conc, 16, 2, (1,1),use_bias=True,name='M_0')
        layerx = tf.layers.flatten(layerx,name='M_1')
        layerx = tf.layers.dense(layerx, 512,name='M_2',use_bias=True)
        layerx = tf.layers.dense(layerx, H,name='M_3',use_bias=True)
    
        #========================================
        layery = tf.layers.conv2d(y_conc, 16, 2, (1,1),name='M_4',use_bias=True)
        layery = tf.layers.flatten(layery,name='M_5')
        layery = tf.layers.dense(layery, 512,name='M_6',use_bias=True)
        layery = tf.layers.dense(layery, H, name='M_7',use_bias=True)
    else:
        layerx = tf.layers.dense(x_conc, 512,name='M_2',use_bias=True)
        layerx = tf.layers.dense(layerx, H,name='M_3',use_bias=True)
        layery = tf.layers.dense(y_conc, 512,name='M_6',use_bias=True)
        layery = tf.layers.dense(layery, H, name='M_7',use_bias=True)
    layer2 = tf.nn.relu(layerx + layery,name='M_8')
    output = tf.layers.dense(layer2, 1,name='M_9',use_bias=False)

    # split in T_xy and T_x_y predictions
    N_samples = tf.shape(x_in)[0]
    T_xy = output[:N_samples]
    T_x_y = output[N_samples:]
    return T_xy, T_x_y


class MINE_supervised:
    def __init__(self, sess, model,batch_size=1, confidence = CONFIDENCE,
                 targeted = False, learning_rate = LEARNING_RATE,
                 max_iterations = MAX_ITERATIONS,
                 initial_const = INITIAL_CONST,
                 boxmin = 0, boxmax = 1, 
                 epsilon=0.3, mine_batch='conv'):
        """
        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. 
        boxmin: Minimum pixel value (default 0).
        boxmax: Maximum pixel value (default 1).
        epsilon: Maximum pixel value can be changed for attack.
        mine_batch: generate batch sample for MINE ('conv', 'random_sampling').

        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.mine_batch = mine_batch
        assert self.batch_size==1, 'Batch size must be 1.'


        shape = (batch_size,image_size,image_size,num_channels)
        
        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.ones(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        
        # the variable we're going to optimize over
        modifier = tf.Variable(np.random.uniform(-epsilon,epsilon,shape).astype('float32'),name='modifier')
        self.modifier = tf.get_variable('modifier',shape,trainable=True, constraint=lambda x: tf.clip_by_value(x, -epsilon, epsilon))
        
        # the resulting image in valid values
        self.newimg = tf.clip_by_value(self.modifier + self.timg,boxmin,boxmax)

        if self.mine_batch == 'random_sampling':
            matrix = tf.random_normal([500,image_size*image_size*num_channels,128],0.,1.0/tf.sqrt(128.))
            self.x_batch = standardized( tf.keras.backend.dot(tf.reshape(self.timg,[image_size*image_size*num_channels]),matrix))
            self.y_batch = standardized( tf.keras.backend.dot(tf.reshape(self.newimg,[image_size*image_size*num_channels]),matrix))
        else:
            self.x_batch = tf.transpose(model.conv1(self.timg),perm=[3,1,2,0])
            self.y_batch = tf.transpose(model.conv1(self.newimg),perm=[3,1,2,0])
        
        T_xy , T_x_y = MiNetwork(self.x_batch, self.y_batch,mine_batch_=self.mine_batch)
        self.MI =  tf.reduce_mean(T_xy, axis=0) - tf.log(tf.reduce_mean(tf.exp(T_x_y)))

        # prediction BEFORE-SOFTMAX of the model
        self.output = model.predict(self.newimg)

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab)*self.output,1)
        other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            self.loss1 = tf.maximum(0.0, other-real+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            self.loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)

        # sum up the losses
        self.loss1_1 = tf.reduce_sum(self.const*self.loss1)
        self.loss = self.loss1_1 - self.MI 
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        m_var = [var for var in tf.global_variables() if 'M_' in var.name]
        optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        self.mi_train = tf.train.AdamOptimizer(0.000015).minimize(-self.MI, var_list=m_var)
        self.train = optimizer.minimize(self.loss, var_list=[self.modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.lamda = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.lamda.append(self.const.assign(self.assign_const))
        self.init = tf.variables_initializer(var_list=[self.modifier]+new_vars)
        self.mi_init = tf.variables_initializer(var_list=m_var)

    def attack(self,imgs, targets):
        """
        Perform the mine-supervised attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            result,m = self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size])
            r.extend(result)
        return np.array(r),m

    def attack_batch(self,imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
   
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl2 = [-1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        
        
        print(o_bestl2)
        # completely reset adam's internal state.
        self.sess.run(self.init)
        self.sess.run(self.mi_init)
        
        batch = imgs[:batch_size]

        batchlab = labs[:batch_size]

        bestl2 = [-1e10]*batch_size
        bestscore = [-1]*batch_size

        # set the variables so that we don't have to send them over again
        self.sess.run(self.setup, {self.assign_timg: batch,
                                   self.assign_tlab: batchlab})

        
        prev = np.inf
        for iteration in range(self.MAX_ITERATIONS):
            self.sess.run(self.lamda,{self.assign_const: CONST})
            # perform the attack
            
            _, l, ls1,scores, nimg = self.sess.run([self.train, self.loss1, self.loss, 
                                               self.output, 
                                               self.newimg])
            #train MINE
            for xx in range(1):
                _, mutual_ins = self.sess.run([self.mi_train,self.MI])
           
            #update constant c
            CONST =(1 - 0.1 * (1/((iteration+1)**0.25)))*CONST + 0.1*l 
            if CONST <0:
                CONST=np.zeros(batch_size)

            
            # print out the losses every 10%
            if iteration%(self.MAX_ITERATIONS//10) == 0:
                print(iteration,self.sess.run((self.loss,self.loss1_1,self.MI)))

            # adjust the best result found so far
            for e,(mutual_in,sc,ii) in enumerate(zip(mutual_ins,scores,nimg)):
                if mutual_in > bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                    bestl2[e] = mutual_in
                    bestscore[e] = np.argmax(sc)
                if mutual_in > o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                    o_bestl2[e] = mutual_in
                    o_bestscore[e] = np.argmax(sc)
                    o_bestattack[e] = ii
        


        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack, o_bestl2
