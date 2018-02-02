import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d

def make_tf_tensors(T, n_timebins, upsample_factor, R, 
                    threshold_a, threshold_dd):
    
    # place holder for input data
    rec_local_tf = tf.placeholder("float32", [T, None])
    template_local_tf = tf.placeholder(
        "float32", [n_timebins, None, upsample_factor])
    spt_tf = tf.placeholder("int32", [None])


    # get number of channels
    n_local_channels = tf.shape(rec_local_tf)[1]

    # get waveforms
    wf_idx = tf.add(spt_tf[:,tf.newaxis],
                    np.arange(-R,R+1)[np.newaxis,:])
    wf_tf = tf.reshape(tf.gather_nd(rec_local_tf,tf.reshape(
        wf_idx, [-1, 1])),[-1, n_timebins, n_local_channels])


    # dot product
    dot_products_tf = tf.matmul(
        tf.reshape(wf_tf,[-1, n_timebins*n_local_channels]), 
        tf.reshape(template_local_tf, [-1, upsample_factor]))
    # norm^2 of the template
    template_norm_tf = tf.reduce_sum(
        np.square(template_local_tf), [0,1])

    # estimate scale
    ahat_tf = tf.divide(dot_products_tf, 
                        tf.expand_dims(template_norm_tf,[0]))
    ahat_max_tf = tf.reduce_max(ahat_tf,1)
    max_idx_tf = tf.argmax(ahat_tf, 1)
    min_bound = (1-threshold_a)
    max_bound = (1+threshold_a)
    ahat_adjusted_tf = tf.minimum(
        tf.maximum(ahat_max_tf, min_bound),max_bound)

    # calculate decrease in L2 norm
    dd_tf = tf.multiply(
        2*tf.multiply(ahat_adjusted_tf, ahat_max_tf) - \
        tf.square(ahat_adjusted_tf), 
        tf.gather(template_norm_tf, max_idx_tf))

    # obtain good locations only
    idx_good = dd_tf > threshold_dd
    dd_good_tf = tf.boolean_mask(dd_tf, idx_good)
    spt_good_tf = tf.boolean_mask(spt_tf, idx_good)
    max_idx_good_tf = tf.boolean_mask(max_idx_tf, idx_good)
    ahat_adjusted_good_tf = tf.boolean_mask(ahat_adjusted_tf, 
                                            idx_good)
    result = (dd_good_tf, spt_good_tf, max_idx_good_tf, 
              ahat_adjusted_good_tf)

    return rec_local_tf, template_local_tf, spt_tf, result


def template_match(rec_local, spt_interest, upsampled_template_local,
                   n_rf, rec_local_tf, template_local_tf, spt_tf, 
                   result
                  ):

    # spt crossing treshold in dd
    with tf.Session() as sess:
        dd, spt, max_idx, ahat = sess.run(
            result,
            feed_dict={rec_local_tf: rec_local,
                       template_local_tf: np.transpose(
                           upsampled_template_local,(2,1,0)),
                       spt_tf: spt_interest})
        
    if dd.shape[0] > 0:
                
        # as we are greedy, accept index with biggest dd only
        dd_long = np.zeros(rec_local.shape[0])
        dd_long[spt] = dd
        idx_good = np.zeros(dd.shape[0], 'bool')
        for j in range(dd.shape[0]):
            tt = spt[j]
            if dd[j] == np.max(dd_long[tt - n_rf: tt + 
                                       n_rf + 1]):
                idx_good[j] = 1

        spt_good = spt[idx_good]
        ahat_good = ahat[idx_good]
        max_idx_good = max_idx[idx_good]

        return spt_good, ahat_good, max_idx_good

    else:
        return np.zeros(0, 'int32'), np.zeros(0, 'int32'), np.zeros(0, 'int32') 