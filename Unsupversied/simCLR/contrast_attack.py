# contrast_attack.py -- It performs MINE-UAE attack.
# author: Chia-Yi Hsu

import sys,time
import tensorflow.compat.v1 as tf
import numpy as np
#import matplotlib.pyplot as plt

MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess
def standardized(img):
    mean, var = tf.nn.moments(tf.convert_to_tensor(img),[1])
    mean = tf.tile(tf.reshape(mean,[-1,1]),[1,tf.shape(img)[1]])
    var = tf.tile(tf.reshape(tf.sqrt(var),[-1,1]),[1, tf.shape(img)[1]])
    img = (img-mean)/var
    return img
def MiNetwork(x_in, y_in):
    H=10
    seed = np.random.randint(0,1000,1)
    
    y_shuffle = tf.gather(y_in, tf.random_shuffle(tf.range(tf.shape(y_in)[0]),seed=seed))
    x_conc = tf.concat([x_in, x_in], axis=0)
    y_conc = tf.concat([y_in, y_shuffle], axis=0)
    
    # propagate the forward pass
    layerx = tf.layers.conv2d(x_conc, 16, 2, (1,1),use_bias=True,name='M_0')
    layerx = tf.layers.flatten(layerx,name='M_1')
    layerx = tf.layers.dense(layerx, 512,name='M_2',use_bias=True)
    layerx = tf.layers.dense(layerx, H,name='M_3',use_bias=True)
    
    #========================================
    layery = tf.layers.conv2d(y_conc, 16, 2, (1,1),name='M_4',use_bias=True)
    layery = tf.layers.flatten(layery,name='M_5')
    layery = tf.layers.dense(layery, 512,name='M_6',use_bias=True)
    layery = tf.layers.dense(layery, H, name='M_7',use_bias=True)
    layer2 = tf.nn.relu(layerx + layery,name='M_8')
    output = tf.layers.dense(layer2, 1,name='M_9',use_bias=False)

    # split in T_xy and T_x_y predictions
    N_samples = tf.shape(x_in)[0]
    T_xy = output[:N_samples]
    T_x_y = output[N_samples:]
    return T_xy, T_x_y
def sparse_random_projection(n_components,n_features):
    density = np.sqrt(1/n_features)
    rng = np.random.mtrand._rand
    if density == 1:
        # skip index generation if totally dense
        components = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        return 1 / np.sqrt(n_components) * components
    else:
        indices = []
        offset = 0
        indptr = [offset]
        for _ in range(n_components):
            # find the indices of the non-zero components for row i
            n_nonzero_i = rng.binomial(n_features, density)
            indices_i = sample_without_replacement(n_features, n_nonzero_i,
                                                   random_state=rng)
            indices.append(indices_i)
            offset += n_nonzero_i
            indptr.append(offset)

        indices = np.concatenate(indices)

        # Among non zero components the probability of the sign is 50%/50%
        data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

        # build the CSR structure by concatenating the rows
        component = sp.csr_matrix((data, indices, indptr),
                                   shape=(n_components, n_features))
        rr = (np.sqrt(1 / density) / np.sqrt(n_components) * component)
        return rr


class MINE_MinMax_unsupervised:
    def __init__(self, sess, model,batch_size=1, confidence = CONFIDENCE,
                 targeted = False, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 boxmin = 0, boxmax = 1,epsilon=0.3):
        

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        
        self.repeat = binary_search_steps >= 10

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

        shape = (batch_size,image_size,image_size,num_channels)
        
        self.uninitialized_vars = []
        for var in tf.all_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                self.uninitialized_vars.append(var)
        
        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.ones(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        # the variable we're going to optimize over
        #modifier = tf.Variable(np.zeros(shape,dtype=np.float32),name='modifier')
        modifier = tf.Variable(np.random.uniform(-epsilon,epsilon,shape).astype('float32'),name='modifier')
        self.modifier = tf.get_variable('modifier',shape,trainable=True, constraint=lambda x: tf.clip_by_value(x, -epsilon, epsilon))
        
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.newimg = tf.clip_by_value(self.modifier + self.timg,0,1)
        '''
        matrix = tf.random_normal([500,image_size*image_size*num_channels,128],0.,1.0/tf.sqrt(128.))
        
        self.x_batch = standardized( tf.keras.backend.dot(tf.reshape(self.timg,[image_size*image_size*num_channels]),matrix))
        self.y_batch = standardized( tf.keras.backend.dot(tf.reshape(self.newimg,[image_size*image_size*num_channels]),matrix))
        '''

        tconv = tf.transpose(model.conv1(self.timg),perm=[3,1,2,0])
        nconv = tf.transpose(model.conv1(self.newimg),perm=[3,1,2,0])
        T_xy , T_x_y = MiNetwork(tconv, nconv)
        #T_xy , T_x_y = MiNetwork(self.x_batch,self.y_batch)
        self.MI =  tf.reduce_mean(T_xy, axis=0) - tf.log(tf.reduce_mean(tf.exp(T_x_y)))
        
        
        real = model.predict(self.timg, self.tlab)
        self.real = real
        #fake = tf.reduce_max(tf.abs(self.timg - self.recon))
        fake = model.predict(self.newimg, self.tlab)
        self.fake = fake
        self.loss1 = tf.maximum(0.0, fake-real)  
        
        # sum up the losses
        self.loss1_1 = tf.reduce_sum(self.const*self.loss1)
        
        #self.loss = self.loss1_1 - self.MI
        self.loss = self.loss1_1 + self.MI
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        m_var = [var for var in tf.global_variables() if 'M_' in var.name]
        #optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
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

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            result, m= self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size])
            r.extend(result)
        return np.array(r), m

    def attack_batch(self, imgs, labs):
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

        # convert to tanh-space
        #imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)
        cc=[]
        ll=[]
        tl=[]
        mm=[]
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        cc.append(CONST[0])
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size

        f_bestl2 = [1e10]*batch_size
        f_bestscore = [-1]*batch_size
        f_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        
        
        print(o_bestl2)

        # completely reset adam's internal state.
        self.sess.run(self.init)
        self.sess.run(self.mi_init)
        '''
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        '''
        self.sess.run(tf.initialize_variables(self.uninitialized_vars))
        
        batch = imgs[:batch_size]

        batchlab = labs[:batch_size]

        bestl2 = [1e10]*batch_size
        bestscore = [-1]*batch_size

        fbestl2 = [1e10]*batch_size
        fbestscore = [-1]*batch_size

        # set the variables so that we don't have to send them over again
        self.sess.run(self.setup, {self.assign_timg: batch,
                                   self.assign_tlab: batchlab})
        


        
        prev = np.inf
        success = False
        for iteration in range(self.MAX_ITERATIONS):
            self.sess.run(self.lamda,{self.assign_const: CONST})
            
            #_, l, ls1,scores, nimg = self.sess.run([self.train, self.loss1, self.loss, self.output, self.newimg])
            _, l, ls1, nimg = self.sess.run([self.train, self.loss1, self.loss, self.newimg])
            
           
            
            for xx in range(20):
            #if iteration ==0:
                _, l2s = self.sess.run([self.mi_train,self.MI])
            mmi = self.sess.run(self.MI)
            
            #if iteration % 10 ==0:
            
            CONST =(1 - 0.1 * (1/((iteration+1)**0.5)))*CONST + 0.1*l 
            if CONST <0:
                CONST=np.zeros(batch_size)


            '''
            if np.all(scores>=-.0001) and np.all(scores <= 1.0001):
                if np.allclose(np.sum(scores,axis=1), 1.0, atol=1e-3):
                    if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                        raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")
            '''
            # print out the losses every 10%
            if iteration%(self.MAX_ITERATIONS//10) == 0:
                print(iteration,self.sess.run((self.loss,self.loss1,self.MI)))

            # check if we should abort search if we're getting nowhere.
            if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                if l > prev*.9999:
                    break
                prev = l

            '''
            # adjust the best result found so far
            for e,(l2,sc,ii) in enumerate(zip(l2s,scores,nimg)):
                if l2 > bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                    bestl2[e] = l2
                    bestscore[e] = np.argmax(sc)
                if l2 > o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                    o_bestl2[e] = l2
                    o_bestscore[e] = np.argmax(sc)
                    o_bestattack[e] = ii
            '''
            
            for e, (l2, ii, fx) in enumerate(zip(l2s,nimg,[l])):
                if fx == 0 and l2 >0:
                    success = True
                    if l2 < bestl2[e] and l2 >0:
                        bestl2[e] = l2
                    if l2 < o_bestl2[e] and l2>0:
                        o_bestl2[e] = l2
                        o_bestattack[e] = ii
                if fx < fbestl2[e]:
                    fbestl2[e] = fx
                if fx < f_bestl2[e] :
                    f_bestl2[e] = fx
                    f_bestattack[e] = ii
       
        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        
        if success :
            return o_bestattack, o_bestl2
        else:
            return f_bestattack, f_bestl2
