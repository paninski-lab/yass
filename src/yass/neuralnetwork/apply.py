import tensorflow as tf
import numpy as np
from collections import defaultdict

from yass.templates.util import strongly_connected_components_iterative


def run_detect_triage_featurize(recordings, sess, x_tf, output_tf,
                                neighbors, rot):
    """Detect spikes using a neural network

    Parameters
    ----------
    recordings: numpy.ndarray (n_observations, n_channels)
        Neural recordings

    x_tf: tf.tensors (n_observations, n_channels)
        placeholder of recording for running tensorflow

    output_tf: tuple of tf.tensors
        a tuple of tensorflow tensors that produce score, spike_index_clear,
        and spike_index_collision

    NND, NNAE, NNT: Neural Net classes
        NND is neural net detector
        NNAE is autoencoder for dimension reduction
        NNT is neural net triage

    neighbors: np.matrix (n_channels, n_neighbors)
       neighboring information

    Returns
    -------
    scores: numpy.ndarray (n_clear_spikes, n_features, n_neigh)
        3D array with the scores for the clear spikes, first dimension is
        the number of spikes, second is the nymber of features and third the
        number of neighboring channels

    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    spike_index_collision: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for collided spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)
    """

    score, spike_index, idx_clean = sess.run(
        output_tf, feed_dict={x_tf: recordings})

    energy = np.ptp(np.matmul(score[:, :, 0], rot.T), axis=1)

    idx_survive = deduplicate(spike_index, energy, neighbors)
    idx_keep = np.logical_and(idx_survive, idx_clean)
    score_clear = score[idx_keep]
    spike_index_clear = spike_index[idx_keep]

    return (score_clear, spike_index_clear, spike_index)


def deduplicate(spike_index, energy, neighbors, w=5):

    # number of data points
    n_data = spike_index.shape[0]

    # separate time and channel info
    TT = spike_index[:, 0]
    CC = spike_index[:, 1]

    # Index counting
    # indices in index_counter[t-1]+1 to index_counter[t] have time t
    T_max = np.max(TT)
    T_min = np.min(TT)
    index_counter = np.zeros(T_max + w + 1, 'int32')
    t_now = T_min
    for j in range(n_data):
        if TT[j] > t_now:
            index_counter[t_now:TT[j]] = j - 1
            t_now = TT[j]
    index_counter[T_max:] = n_data - 1

    # connecting edges
    j_now = 0
    edges = defaultdict(list)
    for t in range(T_min, T_max + 1):

        # time of j_now to index_counter[t] is t
        # j_now to index_counter[t+w] has all index from t to t+w
        max_index = index_counter[t+w]
        cc_temporal_neighs = CC[j_now:max_index+1]

        for j in range(index_counter[t]-j_now+1):
            # check if channels are also neighboring
            idx_neighs = np.where(
                neighbors[cc_temporal_neighs[j],
                          cc_temporal_neighs[j+1:]])[0] + j + 1 + j_now

            # connect edges to neighboring spikes
            for j2 in idx_neighs:
                edges[j2].append(j+j_now)
                edges[j+j_now].append(j2)

        # update j_now
        j_now = index_counter[t]+1

    # Using scc, build connected components from the graph
    idx_survive = np.zeros(n_data, 'bool')
    for scc in strongly_connected_components_iterative(np.arange(n_data),
                                                       edges):
        idx = list(scc)
        idx_survive[idx[np.argmax(energy[idx])]] = 1

    return idx_survive


def fix_indexes(res, idx_local, idx, buffer_size):
    """Fixes indexes from detected spikes in batches

    Parameters
    ----------
    res: tuple
        A tuple with the results from the nnet detector
    idx_local: slice
        A slice object indicating the indices for the data (excluding buffer)
    idx: slice
        A slice object indicating the absolute location of the data
    buffer_size: int
        Buffer size
    """
    score, clear, collision = res

    # get limits for the data (exlude indexes that have buffer data)
    data_start = idx_local[0].start
    data_end = idx_local[0].stop
    # get offset that will be applied
    offset = idx[0].start

    # fix clear spikes
    clear_times = clear[:, 0]
    # get only observations outside the buffer
    idx_not_in_buffer = np.logical_and(clear_times >= data_start,
                                       clear_times <= data_end)
    clear_not_in_buffer = clear[idx_not_in_buffer]
    score_not_in_buffer = score[idx_not_in_buffer]

    # offset spikes depending on the absolute location
    clear_not_in_buffer[:, 0] = (clear_not_in_buffer[:, 0] + offset
                                 - buffer_size)

    # fix collided spikes
    col_times = collision[:, 0]
    # get only observations outside the buffer
    col_not_in_buffer = collision[np.logical_and(col_times >= data_start,
                                                 col_times <= data_end)]
    # offset spikes depending on the absolute location
    col_not_in_buffer[:, 0] = col_not_in_buffer[:, 0] + offset - buffer_size

    return score_not_in_buffer, clear_not_in_buffer, col_not_in_buffer
