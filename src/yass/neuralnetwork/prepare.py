import tensorflow as tf


# FIXME: this isn't used. remove?
def make_whitened_score(score_tf, main_channel_tf, whiten_filter):
    """
    make a tf tesor that outputs whitened score give scor and
    whitening filter

    Parameters
    ----------
    score_tf: tf tensor (n_spikes, n_features, n_neigh)
        a tf tensor holding score object

    main_channel_tf: tf tensor (n_spikes,)
        a tf tensor holding main channel information for each spike

    whiten_filter: numpy.ndarray (n_channels, n_neigh, n_neigh)
        refer above

    Returns
    -------
    tf tensor (n_spikes, n_features, n_neigh)
        a tf tensor holding whitened scores
    """

    # determine which whitening filter is used for each data
    whiten_filter_per_data = tf.gather(whiten_filter, main_channel_tf)

    return tf.matmul(score_tf, whiten_filter_per_data)


# FIXME: this isn't used. remove?
def remove_uncetered(score_tf, waveform_tf, spike_index_tf):
    """
    Given scores remove spatially uncentered spikes. Spatially unceneterd is
    determined by looking at energy (square of L2 norm) in each channel
    and if the main channel does not have the biggest energy, it is uncentered.

    Parameters
    ----------
    score_tf: tf tensor (n_spikes, n_features, n_neigh)
        a tf tensor holding score object

    waveform_tf: tf tensor (n_spikes, n_temporal_length, n_neigh)
        a tf tensor holding waveform object

    spike_index_tf: tf tensor (n_spikes, 2)
        a tf tensor holding spike index.
        The first column is time and the second column is the main channel

    Returns
    -------
    score_keep_tf: tf tensor (n_centered_spikes, n_features, n_neigh)

    wf_keep_tf: tf tensor (n_centered_spikes, n_temporal_length, n_neigh)

    spike_index_keep_tf: tf tensor (n_centered_spikes, 2)

        tf tensors after screening out uncentered ones
    """

    wf_shapes = tf.shape(waveform_tf, out_type='int64')

    abs_wf_tf = tf.abs(waveform_tf)
    energy_tf = tf.reduce_max(abs_wf_tf, 1)
    loc_max = tf.argmax(abs_wf_tf[:, :, 0], 1)

    # it is centered if the energy is highest at the first channel index
    idx_centered = tf.logical_and(tf.equal(tf.argmax(energy_tf, 1), 0),
                                  tf.logical_and(loc_max > 0,
                                                 loc_max < wf_shapes[2]))

    # keep oinly good ones
    wf_keep_tf = tf.boolean_mask(waveform_tf, idx_centered)
    score_keep_tf = tf.boolean_mask(score_tf, idx_centered)
    spike_index_keep_tf = tf.boolean_mask(spike_index_tf, idx_centered)

    return score_keep_tf, wf_keep_tf, spike_index_keep_tf
