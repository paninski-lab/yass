import os
import numpy as np
import tensorflow as tf
import pkg_resources
import progressbar

from .geometry import order_channels_by_distance


class NeuralNetDetector(object):

    def __init__(self, config):
        self.config = config

        C = np.max(np.sum(self.config.neighChannels, 0))

        R1, R2, R3 = self.config.neural_network['nnFilterSize']
        K1, K2, K3 = self.config.neural_network['nnNFilters']

        self.W1 = weight_variable([R1,1,1,K1])
        self.b1 = bias_variable([K1])

        self.W11 = weight_variable([1,1,K1,K2])
        self.b11 = bias_variable([K2])

        self.W2 = weight_variable([1,C,K2,1])
        self.b2 = bias_variable([1])

        # output of ae encoding (1st layer)
        nFeat = config.nFeat
        R = 2*config.spikeSize+1
        self.W_ae = tf.Variable(tf.random_uniform((R, nFeat), -1.0 / np.sqrt(R), 1.0 / np.sqrt(R)))

        self.saver_ae = tf.train.Saver({"W_ae": self.W_ae})
        self.saver = tf.train.Saver({"W1": self.W1, "W11": self.W11, "W2": self.W2, "b1": self.b1, "b11":self.b11, "b2": self.b2})
    def load_w_ae(self):

        with tf.Session() as sess:
        #config = tf.ConfigProto(device_count = {'GPU': 0})
        #with tf.Session(config=config) as sess:
            path_to_aefile = pkg_resources.resource_filename('yass', 'models/{}'.format(self.config.neural_network['aeFilename']))
            self.saver_ae.restore(sess, path_to_aefile)
            return sess.run(self.W_ae)


    
    def get_spikes(self, X):

        # get parameters
        T, C = X.shape
        R1, R2, R3 = self.config.neural_network['nnFilterSize']
        K1, K2, K3 = self.config.neural_network['nnNFilters']
        th = self.config.nnThreshdold
        temporal_window = 3 #self.config.spikeSize
        
        T_small = np.min((10000,T))
        nbatches = int(np.ceil(float(T)/T_small))
        if nbatches == 1:
            buff = 0
        else:
            buff = R1

        # neighboring channel info
        neighChannels = self.config.neighChannels
        geom = self.config.geom
        nneigh = np.max(np.sum(neighChannels, 0))
        c_idx = np.ones((C, nneigh), 'int32')*C
        for c in range(C):
            ch_idx, temp = order_channels_by_distance(c,np.where(neighChannels[c])[0],geom)
            c_idx[c,:ch_idx.shape[0]] = ch_idx

        # NN structures   
        x_tf = tf.placeholder("float", [1, T_small+2*buff, C])
        layer1 = tf.nn.relu( conv2d( tf.expand_dims(x_tf,-1), self.W1 ) + self.b1 )
        layer11 = tf.nn.relu( conv2d( layer1, self.W11 ) + self.b11 )
        zero_added_layer11 = tf.concat( ( tf.transpose(layer11, [2, 0, 1, 3]),
                                          tf.zeros((1,1,T_small+2*buff,K2))
                                        ), axis = 0 )
        temp = tf.transpose( tf.gather( zero_added_layer11, c_idx ), [0,2,3,1,4] )
        temp2 = conv2d_VALID( tf.reshape( temp, [-1, T_small+2*buff, nneigh,K2] ), self.W2 ) + self.b2
        o_layer = tf.transpose( temp2, [2,1,0,3] )

        # get spike times
        #zero_added_output = tf.concat( (o_layer,tf.zeros((1,T_small+2*buff,1,1))), axis = 2)
        #temporal_max = tf.nn.max_pool(zero_added_output, [1,temporal_window,1,1], [1,1,1,1], 'SAME')
        #max_neigh = tf.transpose(tf.squeeze(tf.reduce_max(tf.gather(tf.transpose(temporal_max, [2,1,0,3]), c_idx), axis=1)))
        #result = tf.where(tf.logical_and(o_layer[0,:,:,0] >= max_neigh, o_layer[0,:,:,0] > np.log(th/(1-th))))
        
        temporal_max = max_pool(o_layer, [1,temporal_window,1,1])        
        local_max_idx = tf.where( tf.logical_and( o_layer[0,:,:,0] >= temporal_max[0,:,:,0], 
                                                  o_layer[0,:,:,0] > np.log(th/(1-th))
                                                ) )

        W_ae_conv = tf.expand_dims(tf.expand_dims(self.W_ae,1),1)
        score_tf = conv2d( tf.expand_dims(x_tf,-1), W_ae_conv )  
        energy_tf = tf.squeeze(tf.reduce_sum(tf.square(score_tf),axis=3))
        energy_val = tf.gather_nd(energy_tf, local_max_idx)

        result = tf.concat([local_max_idx, tf.cast(tf.expand_dims(energy_val,-1), 'int64')], 1)

        energy_train_tf = tf.placeholder("float", [T_small+2*buff, C])
        temporal_max_energy = max_pool( 
            tf.expand_dims(tf.expand_dims(energy_train_tf,0),-1), 
            [1,temporal_window,1,1] )
        zero_added_output = tf.concat( (temporal_max_energy,tf.zeros((1,T_small+2*buff,1,1))), axis = 2)   
        max_neigh_energy = tf.transpose( tf.squeeze( tf.reduce_max( 
                    tf.gather( tf.transpose(zero_added_output, [2,1,0,3]), c_idx), axis=1)))
        result2 = tf.where(tf.logical_and(energy_train_tf > 0, energy_train_tf >= max_neigh_energy))

    
        X = np.expand_dims(X, 0)
        index = np.zeros((10000000, 2), 'int32')
        #index = np.zeros((1000000, 4), 'float32')
        count = 0
        with tf.Session() as sess:
        #config = tf.ConfigProto(device_count = {'GPU': 0})
        #with tf.Session(config=config) as sess:
            path_to_nnfile = pkg_resources.resource_filename('yass', 'models/{}'.format(self.config.neural_network['nnFilename']))
            path_to_aefile = pkg_resources.resource_filename('yass', 'models/{}'.format(self.config.neural_network['aeFilename']))

            self.saver.restore(sess, path_to_nnfile)
            self.saver_ae.restore(sess, path_to_aefile)
    
            for j in range(nbatches):
                if buff == 0:
                    index_temp = sess.run(
                        result, feed_dict={x_tf: X})
                elif j == 0:
                    index_temp = sess.run(
                        result, feed_dict={x_tf: X[:, :(T_small+2*buff)]})
                    
                    energy_train = np.zeros((T_small+2*buff,C))
                    energy_train[index_temp[:,0],index_temp[:,1]] = index_temp[:,2] 
                    index_temp = sess.run(
                        result2, feed_dict={energy_train_tf: energy_train })
                    
                    index_temp = index_temp[index_temp[:, 0] < T_small]
                
                elif (T_small*(j+1)+buff) > T:
                    X_temp = X[:, (T_small*j-buff):]
                    zeros_size = T_small+2*buff - X_temp.shape[1]
                    Zerobuff = np.zeros((1, zeros_size, X_temp.shape[2]))
                    index_temp = sess.run(
                        result, feed_dict={x_tf: np.concatenate( (X_temp, Zerobuff ), axis=1)})
                    
                    energy_train = np.zeros((T_small+2*buff,C))
                    energy_train[index_temp[:,0],index_temp[:,1]] = index_temp[:,2] 
                    index_temp = sess.run(
                        result2, feed_dict={energy_train_tf: energy_train })
                    
                    index_temp = index_temp[np.logical_and(
                        index_temp[:, 0] >= buff, index_temp[:, 0] < X_temp.shape[1])]
                    index_temp[:, 0] = index_temp[:, 0] + T_small*j - buff
                else:
                    index_temp = sess.run(
                        result, feed_dict={x_tf: X[:, (T_small*j-buff):(T_small*(j+1)+buff)]})
                    
                    energy_train = np.zeros((T_small+2*buff,C))
                    energy_train[index_temp[:,0],index_temp[:,1]] = index_temp[:,2] 
                    index_temp = sess.run(
                        result2, feed_dict={energy_train_tf: energy_train })
                    
                    index_temp = index_temp[np.logical_and(
                        index_temp[:, 0] >= buff, index_temp[:, 0] < buff+T_small)]
                    index_temp[:, 0] = index_temp[:, 0] + T_small*j - buff

                index[count:(count+index_temp.shape[0])] = index_temp
                count += index_temp.shape[0]
                
        index = index[:count]
        index = np.concatenate((index, np.ones((count,1),'int32')), axis = 1)
        return index

    def train_ae(self, x_train, y_train, nn_name):

        ndata, n_input = x_train.shape
        n_hidden = self.config.nFeat

        x_ = tf.placeholder("float", [None, n_input])
        y_ = tf.placeholder("float", [None, n_input])

        h = tf.matmul(x_, self.W_ae)

        Wo = tf.transpose(self.W_ae)
        y_tf = tf.matmul(h, Wo)

        cross_entropy = -tf.reduce_sum(y_*tf.log(y_tf))
        meansq = tf.reduce_mean(tf.square(y_-y_tf))
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(meansq)

        niter = 2000
        bar = progressbar.ProgressBar(maxval=niter)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for i in range(0, niter):
                sess.run(train_step, feed_dict={x_: x_train, y_: y_train})
                bar.update(i+1)
            self.saver_ae.save(sess, os.path.join(
                self.config.root, nn_name))
        bar.finish()
        
    def train_detector(self, x_train, y_train, nn_name):

        # iteration info
        niter = int(self.config.neural_network['nnIteration'])
        nbatch = int(self.config.neural_network['nnBatch'])        

        # get parameters
        ndata, T, C = x_train.shape
        R1, R2, R3 = self.config.neural_network['nnFilterSize']
        K1, K2, K3 = self.config.neural_network['nnNFilters']

        x_tf = tf.placeholder("float", [nbatch, T, C])
        y_tf = tf.placeholder("float", [nbatch])
        
        layer1 = tf.nn.relu( conv2d_VALID( tf.expand_dims(x_tf,-1), self.W1 ) + self.b1 )
        layer11 = tf.nn.relu( conv2d( layer1, self.W11 ) + self.b11 )
        o_layer = tf.squeeze( conv2d_VALID( layer11, self.W2 ) + self.b2 )

        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=o_layer, labels=y_tf) )

        weights = tf.trainable_variables()
        l2_regularizer = tf.contrib.layers.l2_regularizer(
            scale= self.config.neural_network['nnL2RegScale'])
        regularization_penalty = tf.contrib.layers.apply_regularization(
            l2_regularizer, weights)
        regularized_loss = cross_entropy + regularization_penalty
        
        train_step = tf.train.AdamOptimizer(
            self.config.neural_network['nnTrainStepSize']).minimize(regularized_loss)
        
        # training
        bar = progressbar.ProgressBar(maxval=niter)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for i in range(0, niter):
                idx_batch = np.random.choice(ndata, nbatch, replace=False)
                sess.run(train_step, feed_dict={x_tf: x_train[idx_batch], y_tf: y_train[idx_batch]})                
                bar.update(i+1)
            self.saver.save(sess, os.path.join(self.config.root, nn_name))
        bar.finish()

class NeuralNetTriage(object):
    
    def __init__(self,config):
        
        self.config = config
        
        self.nneigh = np.max(np.sum(config.neighChannels, 0))
        D = (2*config.spikeSize+1)*self.nneigh
        ncells = config.neural_network['nnTriageFilterSize']
        
        W1 = weight_variable([D,ncells[0]])
        W2 = weight_variable([ncells[0],ncells[1]])
        W3 = weight_variable([ncells[1],1])

        b1 = bias_variable([ncells[0]])
        b2 = bias_variable([ncells[1]])
        b3 = bias_variable([1])

        self.x_tf = tf.placeholder("float", [None, D])
        layer1 = tf.nn.relu(tf.add(tf.matmul(self.x_tf, W1), b1))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2))
        self.o_layer = tf.squeeze(tf.add(tf.matmul(layer2, W3), b3))
        self.tf_prob = tf.sigmoid(self.o_layer)
        
        self.ckpt_loc = pkg_resources.resource_filename('yass', 'models/{}'.format(self.config.neural_network['nnTriageFilename']))
        self.saver_triagenet = tf.train.Saver({"W1": W1,"W2": W2,"W3": W3,"b1": b1,"b2": b2,"b3": b3})
    
    def nn_triage(self, wf, th):
        
        nneigh = self.nneigh
        n,R,C = wf.shape

        if C < nneigh:
            wf = np.concatenate( (wf,np.zeros((n,R,nneigh-C))), axis = 2)

        with tf.Session() as sess:
        #config = tf.ConfigProto(device_count = {'GPU': 0})
        #with tf.Session(config=config) as sess:
            self.saver_triagenet.restore(sess, self.ckpt_loc)
            
            pp = sess.run(self.tf_prob, feed_dict={self.x_tf: np.reshape(wf,(n,-1))})
            return pp > th


    def train_triagenet(self, x_train, y_train, nn_name):

        ndata, T, C = x_train.shape
        
        # iteration info
        niter = int(self.config.neural_network['nnIteration'])
        nbatch = int(self.config.neural_network['nnBatch'])        
    
        y_tf = tf.placeholder("float", [nbatch])

        tf_prob = tf.sigmoid(self.o_layer)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.o_layer, labels=y_tf))

        weights = tf.trainable_variables()
        l2_penalty = self.config.neural_network['nnL2RegScale']
        l2_regularizer = tf.contrib.layers.l2_regularizer(
            scale=l2_penalty)
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
        regularized_loss = cross_entropy + regularization_penalty
        
        TrainStepSize = self.config.neural_network['nnTrainStepSize']
        train_step = tf.train.AdamOptimizer(TrainStepSize).minimize(regularized_loss)

        bar = progressbar.ProgressBar(maxval=niter)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for i in range(0, niter):
                idx_batch = np.random.choice(ndata, nbatch, replace=False)

                sess.run(train_step, feed_dict={self.x_tf: np.reshape(x_train[idx_batch],[nbatch,-1]), y_tf:  y_train[idx_batch]})                
                bar.update(i+1)
            self.saver_triagenet.save(sess, os.path.join(self.config.root, nn_name))
        bar.finish()
        
            
def weight_variable(shape, varName=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=varName)

def bias_variable(shape, varName=None):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=varName)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def conv2d_VALID(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')            

def max_pool(x, W):
    return tf.nn.max_pool(x, W, strides=[1, 1, 1, 1], padding='SAME')            
