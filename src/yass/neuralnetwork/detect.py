import tensorflow as tf
import pkg_resources
import numpy as np

from .nndetector import NeuralNetDetector
from .nntriage import NeuralNetTriage
from .remove import remove_duplicate_spikes_by_energy
from .score import get_score

from ..geometry import order_channels_by_distance

def nn_detection(X, T_batch, buff, neighChannels, geom, 
               n_features, temporal_window, th_detect, th_triage, 
               nnd, nnt):

    T, C = X.shape
    
    T_small = np.min((T_batch,T))
    nbatches = int(np.ceil(float(T)/T_small))
    if nbatches == 1:
        buff = 0

    # neighboring channel info
    nneigh = np.max(np.sum(neighChannels, 0))
    c_idx = np.ones((C, nneigh), 'int32')*C
    for c in range(C):
        ch_idx, temp = order_channels_by_distance(c,np.where(neighChannels[c])[0],geom)
        c_idx[c,:ch_idx.shape[0]] = ch_idx


    # input 
    x_tf = tf.placeholder("float", [T_small+2*buff, C])

    # detect spike index
    local_max_idx_tf = nnd.get_spikes(x_tf, T_small+2*buff, nneigh, c_idx, temporal_window, th_detect)
    
    # get score train
    score_train_tf = nnd.get_score_train(x_tf)
    
    # get energy for detected index
    energy_tf = tf.reduce_sum(tf.square(score_train_tf),axis=2)
    energy_val_tf = tf.gather_nd(energy_tf, local_max_idx_tf)
    
    # get triage probability
    triage_prob_tf = nnt.triage_prob(x_tf, T_small+2*buff, nneigh, c_idx)
    
    # gather all results above
    result = (local_max_idx_tf, score_train_tf, energy_val_tf, triage_prob_tf)
        
    # remove duplicates            
    energy_train_tf = tf.placeholder("float", [T_small+2*buff, C])
    spike_index_tf = remove_duplicate_spikes_by_energy(energy_train_tf, T_small+2*buff, c_idx, temporal_window)        
    
    # get score
    score_train_placeholder = tf.placeholder("float", [T_small+2*buff, C, n_features])
    spike_index_clear_tf = tf.placeholder("int64", [None, 2])
    score_tf = get_score(score_train_placeholder, spike_index_clear_tf, T_small+2*buff, n_features, c_idx)
    
    ###############################
    # get values of above tensors #
    ###############################
    
    spike_index_clear_all = np.zeros((10000000, 2), 'int32')
    spike_index_collision_all = np.zeros((10000000, 2), 'int32')
    score_all = np.zeros((10000000, n_features, nneigh))

    with tf.Session() as sess:

        nnd.saver.restore(sess, nnd.path_to_detector_model)
        nnd.saver_ae.restore(sess, nnd.path_to_ae_model)
        nnt.saver.restore(sess, nnt.path_to_triage_model)

        counter_clear = 0
        counter_collision = 0
        for j in range(nbatches):
            if buff == 0:
                X_batch = X

                T_min = 0
                T_max = T_small
                t_add = 0

            elif j == 0:
                X_batch = X[:(T_small+2*buff)]

                T_min = 0
                T_max = T_small
                t_add = 0

            elif (T_small*(j+1)+buff) > T:
                X_temp = X[(T_small*j-buff):]
                zeros_size = T_small+2*buff - X_temp.shape[0]
                Zerobuff = np.zeros((zeros_size, X_temp.shape[1]))
                X_batch = np.concatenate((X_temp, Zerobuff))

                T_min = buff
                T_max = X_temp.shape[1]
                t_add = T_small*j - buff

            else:
                X_batch = X[(T_small*j-buff):(T_small*(j+1)+buff)]

                T_min = buff
                T_max = buff+T_small
                t_add = T_small*j - buff
            
            local_max_idx, score_train, energy_val, triage_prob  = sess.run(result, feed_dict={x_tf: X_batch})

            energy_train = np.zeros((T_small+2*buff,C))
            energy_train[local_max_idx[:,0],local_max_idx[:,1]] = energy_val
            spike_index = sess.run(spike_index_tf, feed_dict={energy_train_tf: energy_train})
            spike_index = spike_index[np.logical_and(
                        spike_index[:, 0] >= T_min, spike_index[:, 0] < T_max)]

            idx_clean = triage_prob[spike_index[:,0],spike_index[:,1]] > th_triage

            spike_index_clear = spike_index[idx_clean]
            spike_index_collision = spike_index[~idx_clean]
            
            score = sess.run(score_tf, feed_dict={score_train_placeholder:score_train, spike_index_clear_tf:spike_index_clear})
            
            spike_index_clear[:,0] = spike_index_clear[:,0] + t_add
            spike_index_collision[:,0] = spike_index_collision[:,0] + t_add

            n_clean = spike_index_clear.shape[0]
            n_col = spike_index_collision.shape[0]

            spike_index_clear_all[counter_clear:(counter_clear+n_clean)] = spike_index_clear
            spike_index_collision_all[counter_collision:(counter_collision+n_col)] = spike_index_collision
            score_all[counter_clear:(counter_clear+n_clean)] = score
            
            counter_clear += n_clean
            counter_collision += n_col

    spike_index_clear_all = spike_index_clear_all[:counter_clear]
    spike_index_collision_all = spike_index_collision_all[:counter_collision]
    score_all = score_all[:counter_clear]
    
    return spike_index_clear_all, spike_index_collision_all, score_all