"""
Whitening functions
"""

import numpy as np

from yass.geometry import n_steps_neigh_channels, order_channels_by_distance

def matrix(recording, channel_index, spike_size):
    """Spatial whitening filter for time series
    [How is this different from the other method?]

    Parameters
    ----------
    recording: np.array
        T x C numpy array, where T is the number of time samples and
        C is the number of channels

    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        whitening matrix
    """
    # get all necessary parameters from param
    n_observations, n_channels = recording.shape
    R = spike_size*2 + 1
    th = 4
    n_neigh = channel_index.shape[1]

    # masked recording
    spikes_rec = np.ones((n_observations, n_channels))
    for i in range(0, n_channels):
        idxCrossing = np.where(recording[:, i] < -th)[0]
        idxCrossing = idxCrossing[np.logical_and(
            idxCrossing >= (R+1), idxCrossing <= (n_observations-R-1))]
        spike_time = idxCrossing[np.logical_and(
            recording[idxCrossing, i] <= recording[idxCrossing-1, i],
            recording[idxCrossing, i] <= recording[idxCrossing+1, i])]

        # the portion of recording where spikes present is set to nan
        for j in np.arange(-spike_size, spike_size+1):
            spikes_rec[spike_time + j, i] = 0

    # get covariance matrix
    blanked_rec = recording*spikes_rec
    M = np.matmul(blanked_rec.transpose(), blanked_rec) / \
        np.matmul(spikes_rec.transpose(), spikes_rec)

    # since recording is standardized recording, covaraince = correlation
    invhalf_var = np.diag(np.power(np.diag(M), -0.5))
    M = np.matmul(np.matmul(invhalf_var, M), invhalf_var)

    # get localized whitening filter
    whiten_filter = np.zeros((n_channels, n_neigh, n_neigh), 'float32')
    for c in range(0, n_channels):
        ch_idx = channel_index[c][channel_index[c] < n_channels]
        nneigh_c = ch_idx.shape[0]

        V, D, _ = np.linalg.svd(M[ch_idx, :][:, ch_idx])
        eps = 1e-6
        Epsilon = np.diag(1/np.power((D + eps), 0.5))
        Q_small = np.matmul(np.matmul(V, Epsilon), V.transpose())
        whiten_filter[c][:nneigh_c][:, :nneigh_c] = Q_small

    return whiten_filter


def score(scores, main_channel, whiten_filter):
    """?

    Parameters
    ----------
    ?

    Returns
    -------
    ?
    """
    n_data, n_features, n_neigh = scores.shape
    n_channels = whiten_filter.shape[0]

    whitened_scores = np.zeros(scores.shape)
    for c in range(n_channels):
        idx = main_channel == c
        whitened_scores_c = np.matmul(
            np.reshape(scores[idx], [-1, n_neigh]), whiten_filter[c])
        whitened_scores[idx] = np.reshape(whitened_scores_c,
                                         [-1, n_features, n_neigh])

    return whitened_scores
