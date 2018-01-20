"""
PCA for multi-channel recordings, used for dimensionality reduction
when using threshold detector
"""
import logging


import numpy as np

from ..geometry import ordered_neighbors

# TODO: improve documentation: look for (?)
# TODO: remove batching logic and update preprocessor to run this channel
# by channel instead of using multi-channel operations
# TODO: can this be a single-channel operation? that way we can parallelize
# by channel

logger = logging.getLogger(__name__)


def suff_stat(recordings, spike_index, spike_size):
    """
    Get PCA SS matrix per recording channel

    Parameters
    ----------
    recordings: np.ndarray (n_observations, n_channels)
        Multi-channel recordings
    spike_index: np.ndarray (number of spikes, 2)
        Spike indexes as returned from the threshold detector
    spike_size: int
        Spike size

    Returns
    -------
    numpy.ndarray
        3D array (?)
    numpy.ndarray
        1D array, with n_channels entries, the ith entry contains the number
        of spikes found in the ith channel
    """
    # column ids for index matrix
    SPIKE_TIME, MAIN_CHANNEL = 0, 1

    n_obs, n_channels = recordings.shape
    window_idx = range(-spike_size, spike_size+1)
    window_size = len(window_idx)

    pca_suff_stat = np.zeros((window_size, window_size, n_channels))
    spikes_per_channel = np.zeros(n_channels, 'int32')

    # iterate over every channel
    for c in range(n_channels):
        # get spikes times for the current channel
        channel_spike_times = spike_index[
            spike_index[:, MAIN_CHANNEL] == c, SPIKE_TIME]
        channel_spike_times = channel_spike_times[np.logical_and(
            (channel_spike_times > spike_size),
            (channel_spike_times < n_obs-spike_size-1))]

        channel_spikes = len(channel_spike_times)

        # create zeros matrix (window size x number of spikes for this channel)
        wf_temp = np.zeros((window_size, channel_spikes))

        # iterate over the window size
        for j in range(window_size):
            # fill in recording values for each spike time
            wf_temp[j, :] = recordings[channel_spike_times + window_idx[j], c]

        pca_suff_stat[:, :, c] = np.matmul(wf_temp, wf_temp.T)

        spikes_per_channel[c] = channel_spikes

    return pca_suff_stat, spikes_per_channel


def project(ss, spikes_per_channel, n_features, neighbors):
    """
    Get PCA projection matrix per channel

    Parameters
    ----------
    ss: matrix
        SS matrix as returned from get_pca_suff_stat
    spikes_per_channel: array
        Number of spikes per channel
    n_features: int
        Number of features
    neighbors: matrix
        Neighbors matrix

    Returns
    -------
    numpy.ndarray
        3D array (window_size, n_features, n_channels)
    """
    window_size, _, n_channels = ss.shape
    # allocate rotation matrix for each channel
    rot = np.zeros((window_size, n_features, n_channels))

    ss_all = np.sum(ss, 2)
    w, v = np.linalg.eig(ss_all)
    rot_all = v[:, np.argsort(w)[window_size:(window_size-n_features-1):-1]]

    for c in range(n_channels):
        if spikes_per_channel[c] <= window_size:
            if np.sum(spikes_per_channel[neighbors[c, :]]) <= window_size:
                rot[:, :, c] = rot_all
            else:
                w, v = np.linalg.eig(np.sum(ss[:, :, neighbors[c, :]], 2))
                rot[:, :, c] = v[:, np.argsort(
                    w)[window_size:(window_size-n_features-1):-1]]
        else:
            w, v = np.linalg.eig(ss[:, :, c])
            rot[:, :, c] = v[:, np.argsort(
                w)[window_size:(window_size-n_features-1):-1]]

    return rot


def score(waveforms, spike_index, rot, neighbors, geom):
    """Reduce spikes dimensionality with a PCA rotation matrix

    Parameters
    ----------
    waveforms: numpy.ndarray (n_spikes, temporal_window, n_channels)
        Waveforms to score
    spike_index: numpy.ndarray (n_spikes, 2)
        Spike indexes as returned from the threshold detector
    rot: numpy.ndarray (temporal_window, n_features, n_channels)
        PCA rotation matrix
    neighbors: numpy.ndarray (n_channels, n_channels)
        Neighbors matrix
    geom: numpy.ndarray (n_channels, 2)
        Channels location matrix

    Returns
    -------
    [n_spikes, n_features_per_channel, n_neighboring_channels]
        Scores for evert spike
    """
    # TODO: check dimensions

    # TODO: this should be done by the project function
    rot_ = np.transpose(rot)
    sp = np.transpose(waveforms)

    # compute scores for every spike
    score = np.transpose(np.matmul(rot_, sp), (2, 1, 0))

    # for every spike, get the score only for the neighboring channels
    ord_neighbors, channel_features = ordered_neighbors(geom, neighbors)
    spikes, temporal_features, n_channels = score.shape

    print('score shape', score.shape)

    score_neigh = np.zeros((spikes, temporal_features, channel_features))

    logger.info('Scoring {} spikes...'.format(spikes))

    # for every spike...
    for i in range(spikes):

        # get main channel
        main_channel = spike_index[i, 1]

        # get the ordered neighbors for the main channel
        current_neigh = ord_neighbors[main_channel]

        # assign the scores for those channels to the matrix
        score_neigh[i, :, :len(current_neigh)] = score[i][:, current_neigh]

    return score_neigh
