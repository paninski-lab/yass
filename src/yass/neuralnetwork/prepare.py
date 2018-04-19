import tensorflow as tf

from yass.neuralnetwork import NeuralNetDetector, NeuralNetTriage, AutoEncoder


def prepare_nn(channel_index, whiten_filter,
               threshold_detect, threshold_triage,
               detector_filename,
               autoencoder_filename,
               triage_filename):

    """Prepare neural net tensors in advance. This is to effciently run
    neural net with batch processor as we don't have to recreate tf
    tensors in every batch

    Parameters
    ----------
    channel_index: np.array (n_channels, n_neigh)
        Each row indexes its neighboring channels.
        For example, channel_index[c] is the index of
        neighboring channels (including itself)
        If any value is equal to n_channels, it is nothing but
        a space holder in a case that a channel has less than
        n_neigh neighboring channels

    whiten_filter: numpy.ndarray (n_channels, n_neigh, n_neigh)
        whitening matrix such that whiten_filter[c] is the whitening
        filter of channel c and its neighboring channel determined from
        channel_index.

    threshold_detect: int
        threshold for neural net detection

    threshold_triage: int
        threshold for neural net triage

    detector_filename: str
        location of trained neural net detectior

    autoencoder_filename: str
        location of trained neural net autoencoder

    triage_filename: str
        location of trained neural net triage

    Returns
    -------
    x_tf: tf.tensors (n_observations, n_channels)
        placeholder of recording for running tensorflow

    output_tf: tuple of tf.tensors
        a tuple of tensorflow tensors that produce score, spike_index_clear,
        and spike_index_collision

    NND: class
        an instance of class, NeuralNetDetector

    NNT: class
        an instance of class, NeuralNetTriage
    """

    # placeholder for input recording
    x_tf = tf.placeholder("float", [None, None])

    # load Neural Net's
    NND = NeuralNetDetector(detector_filename)
    NNAE = AutoEncoder(autoencoder_filename)
    NNT = NeuralNetTriage(triage_filename)

    # make spike_index tensorflow tensor
    spike_index_tf_all = NND.make_detection_tf_tensors(x_tf,
                                                       channel_index,
                                                       threshold_detect)

    # remove edge spike time
    spike_index_tf = remove_edge_spikes(x_tf, spike_index_tf_all,
                                        NND.filters_dict['size'])

    # make waveform tensorflow tensor
    waveform_tf = make_waveform_tf_tensor(x_tf,
                                          spike_index_tf,
                                          channel_index,
                                          NND.filters_dict['size'])

    # make score tensorflow tensor from waveform
    score_tf = NNAE.make_score_tf_tensor(waveform_tf)

    # run neural net triage
    nneigh = NND.filters_dict['n_neighbors']
    idx_clean = NNT.triage_wf(waveform_tf[:, :, :nneigh], threshold_triage)

    # gather all output tensors
    output_tf = (score_tf, spike_index_tf, idx_clean)

    return x_tf, output_tf, NND, NNAE, NNT


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


def make_waveform_tf_tensor(x_tf, spike_index_tf,
                            channel_index, waveform_length):
    """
    It produces a tf tensor holding waveforms given recording and spike index.
    It does not hold waveforms on all channels but channels around their main
    channels specified in channel_index

    Parameters
    ----------
    x_tf: tf.tensors (n_observations, n_channels)
        placeholder of recording for running tensorflow

    spike_index_tf: tf tensor (n_spikes, 2)
        a tf tensor holding spike index.
        The first column is time and the second column is the main channel

    channel_index: np.array (n_channels, n_neigh)
        refer above

    waveform_length: int
        temporal length of waveform

    Returns
    -------
    tf tensor (n_spikes, waveform_length, n_neigh)
    """
    # get waveform temporally
    R = int((waveform_length-1)/2)
    spike_time = tf.expand_dims(spike_index_tf[:, 0], -1)
    temporal_index = tf.expand_dims(tf.range(-R, R+1), 0)
    wf_temporal = tf.add(spike_time, temporal_index)

    # get waveform spatially
    nneigh = channel_index.shape[1]
    wf_spatial = tf.gather(channel_index, spike_index_tf[:, 1])

    wf_temporal_expand = tf.expand_dims(
        tf.expand_dims(wf_temporal, -1), -1)
    wf_spatial_expand = tf.expand_dims(
        tf.expand_dims(wf_spatial, 1), -1)

    wf_idx = tf.concat((tf.tile(wf_temporal_expand, (1, 1, nneigh, 1)),
                        tf.tile(wf_spatial_expand,
                                (1, waveform_length, 1, 1))), 3)

    # temproal length of recording
    T = tf.shape(x_tf)[0]
    x_tf_zero_added = tf.concat([x_tf, tf.zeros((T, 1))], axis=1)

    return tf.gather_nd(x_tf_zero_added, wf_idx)


def remove_edge_spikes(x_tf, spike_index_tf, waveform_length):
    """
    It moves spikes at edge times.

    Parameters
    ----------
    x_tf: tf.tensors (n_observations, n_channels)
        placeholder of recording for running tensorflow

    spike_index_tf: tf tensor (n_spikes, 2)
        a tf tensor holding spike index.
        The first column is time and the second column is the main channel

    waveform_length: int
        temporal length of waveform

    Returns
    -------
    tf tensor (n_spikes, 2)
    """

    R = int((waveform_length-1)/2)
    min_spike_time = R
    max_spike_time = tf.shape(x_tf)[0] - R

    idx_middle = tf.logical_and(spike_index_tf[:, 0] > min_spike_time,
                                spike_index_tf[:, 0] < max_spike_time)

    return tf.boolean_mask(spike_index_tf, idx_middle)
