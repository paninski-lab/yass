import tensorflow as tf
import numpy as np


def make_tf_tensors(T, waveform_size, n_shifts,
                    threshold_a, threshold_d):
    """
    Make tensorflow tensors necessary for deconvolution given parameters

    Parameters
    ----------

    T: int
        The temporal length of recording

    waveform_size: int
        Temporal length of each templte

    n_shifts: int
        number of shifted templates

    threshold_a: int
        threhold on waveform scale when fitted to template

    threshold_d: int
        threshold on decrease in l2 norm of recording after
        subtracting a template


    returns
    -------
    rec_local_tf: tensorflow tensor (T, n_local_channels)
        placeholder for recording with a subset of channels

    template_local_tf: tensorflow tensor (waveform_size,
        n_local_channels, n_shifts)
        placeholder for shifted templates on a subset of channels

    spt_tf: tensorflow tensor (n_spikes)
        placeholder for spike times

    result: tuple (dd_good_tf, spt_good_tf, max_idx_good_tf,
                   ahat_good_tf)
        contains information of deconvolved spikes

        dd_good_tf: tensorflow tensor (n_good_spikes)
            decrease in objective function
        spt_good_tf: tensorflow tensor (n_good_spikes)
            spike times
        max_idx_good_tf: tensorflow tensor (n_good_spikes)
            which shifted template gives the best fit
        ahat_good_tf: tensorflow tensor (n_good_spikes)
            estimated scale of each spike relative to template
    """

    # place holder for input data
    rec_local_tf = tf.placeholder("float32", [T, None])
    template_local_tf = tf.placeholder(
        "float32", [waveform_size, None, n_shifts])
    spt_tf = tf.placeholder("int32", [None])

    # get number of channels
    n_local_channels = tf.shape(rec_local_tf)[1]

    # get waveforms
    R = int((waveform_size - 1)/2)
    wf_idx = tf.add(spt_tf[:, tf.newaxis],
                    np.arange(-R, R+1)[np.newaxis, :])
    wf_tf = tf.reshape(tf.gather_nd(rec_local_tf, tf.reshape(
        wf_idx, [-1, 1])), [-1, waveform_size, n_local_channels])

    # dot product between waveform and template
    # dot_products_tf has size of n_spikes x n_shifts
    dot_products_tf = tf.matmul(
        tf.reshape(wf_tf, [-1, waveform_size*n_local_channels]),
        tf.reshape(template_local_tf, [-1, n_shifts]))

    # best fit shift
    max_idx_tf = tf.argmax(dot_products_tf, 1)
    dot_products_max_tf = tf.reduce_max(dot_products_tf, 1)

    # norm^2 of the template
    template_norm_tf = tf.reduce_sum(
        np.square(template_local_tf), [0, 1])

    best_norm_tf = tf.gather(template_norm_tf,
                             max_idx_tf)
    ahat1_tf = tf.divide(dot_products_max_tf,
                         best_norm_tf)

    # best fit is what maximizes the scale
    min_bound = (1-threshold_a)
    max_bound = (1+threshold_a)
    ahat_tf = tf.minimum(
        tf.maximum(ahat1_tf, min_bound), max_bound)

    norm_fit = tf.multiply(tf.square(ahat_tf), best_norm_tf)
    dd_tf = 2*tf.multiply(ahat_tf, dot_products_max_tf) - norm_fit

    # obtain good locations only
    idx_good = dd_tf > threshold_d*norm_fit
    dd_good_tf = tf.boolean_mask(dd_tf,
                                 idx_good)
    spt_good_tf = tf.boolean_mask(spt_tf,
                                  idx_good)
    max_idx_good_tf = tf.boolean_mask(max_idx_tf,
                                      idx_good)
    ahat_good_tf = tf.boolean_mask(ahat_tf,
                                   idx_good)
    result = (dd_good_tf, spt_good_tf, max_idx_good_tf, ahat_good_tf)

    return rec_local_tf, template_local_tf, spt_tf, result


def template_match(rec_local, spt, upsampled_template_local,
                   n_rf, rec_local_tf, template_local_tf, spt_tf,
                   result
                   ):
    """
    Run template match

    Parameters
    ----------

    rec_local:  numpy.ndarray (T, n_local_channels)
        recording with a subset of channels

    spt: numpy.ndarray (n_spikes)
        spike time

    upsampled_template_local: numpy.ndarray (n_shifts,
        n_local_channels, waveform_size)
        shifted templates on a subset of channels

    n_rf: int
        number of timebins for refractory period violation

    rec_local_tf, template_local_tf,
    spt_tf, result: tensorflow tensor
        output of make_tf_tensors function

    returns
    -------
    spt_good: numpy.ndarray (n_good_spikes)
        deconvolved spike times
    ahat_good: numpy.ndarray (n_good_spikes)
        scale of each deconvolved spike relative to template
    max_idx_good: numpy.ndarray (n_good_spikes)
        index for shifted template chosen for each spike
    """

    # spt crossing treshold in dd
    with tf.Session() as sess:
        dd, spt, max_idx, ahat = sess.run(
            result,
            feed_dict={rec_local_tf: rec_local,
                       template_local_tf: np.transpose(
                           upsampled_template_local, (2, 1, 0)),
                       spt_tf: spt})

    if dd.shape[0] > 0:

        # among deconvolved spikes in the first step, obtain only
        # spikes with maximal decrease in objective function
        # within refractory period
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
