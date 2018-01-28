"""
Whitening functions
"""

import numpy as np

from ..geometry import n_steps_neigh_channels, order_channels_by_distance


# TODO: missing documentation
# TODO: add comments to the code to understand what's going on
# TODO: document why we need different methods for whitening when using
# threshold detection vs nn detection
# TODO: how to apply this in batches


def matrix(ts, neighbors, spike_size):
    """
    Compute spatial whitening matrix for time series, used only in threshold
    detection

    Parameters
    ----------
    ts: numpy.ndarray (n_observations, n_channels)
        Recordings
    neighbors: numpy.ndarray (n_channels, n_channels)
        Boolean numpy 2-D array where a i, j entry is True if i is considered
        neighbor of j
    spike_size: int
        Spike size

    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        whitening matrix
    """
    # get all necessary parameters from param
    [T, C] = ts.shape
    R = spike_size*2 + 1
    th = 4
    neighChannels = n_steps_neigh_channels(neighbors, steps=2)

    chanRange = np.arange(0, C)
    spikes_rec = np.ones(ts.shape)

    for i in range(0, C):
        idxCrossing = np.where(ts[:, i] < -th)[0]
        idxCrossing = idxCrossing[np.logical_and(
            idxCrossing >= (R+1), idxCrossing <= (T-R-1))]
        spike_time = idxCrossing[
            np.logical_and(ts[idxCrossing, i] <= ts[idxCrossing-1, i],
                           ts[idxCrossing, i] <= ts[idxCrossing+1, i])]

        # the portion of recording where spikes present is set to nan
        for j in np.arange(-spike_size, spike_size+1):
            spikes_rec[spike_time + j, i] = 0

    blanked_rec = ts*spikes_rec
    M = np.matmul(blanked_rec.transpose(), blanked_rec) / \
        np.matmul(spikes_rec.transpose(), spikes_rec)
    invhalf_var = np.diag(np.power(np.diag(M), -0.5))
    M = np.matmul(np.matmul(invhalf_var, M), invhalf_var)
    Q = np.zeros((C, C))

    for c in range(0, C):
        ch_idx = chanRange[neighChannels[c, :]]
        V, D, _ = np.linalg.svd(M[ch_idx, :][:, ch_idx])
        eps = 1e-6
        Epsilon = np.diag(1/np.power((D + eps), 0.5))
        Q_small = np.matmul(np.matmul(V, Epsilon), V.transpose())
        Q[c, ch_idx] = Q_small[ch_idx == c, :]

    return Q.transpose()


# TODO: remove
def apply(ts, neighbors, spike_size, Q=None):
    """Apply spatial whitening

    Parameters
    ----------
    ts: numpy.ndarray (n_observations, n_channels)
        Recordings
    neighbors: numpy.ndarray (n_channels, n_channels)
        Boolean numpy 2-D array where a i, j entry is True if i is considered
        neighbor of j
    spike_size: int
        Spike size

    Returns
    -------
    numpy.ndarray (n_observations, n_channels)
        Whitened data
    """
    return np.matmul(ts, Q)


def matrix_localized(ts, neighbors, geom, spike_size):
    """Spatial whitening filter for time series
    [How is this different from the other method?]

    Parameters
    ----------
    ts: np.array
        T x C numpy array, where T is the number of time samples and
        C is the number of channels

    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        whitening matrix
    """
    # get all necessary parameters from param
    [T, C] = ts.shape
    R = spike_size*2 + 1
    th = 4
    nneigh = np.max(np.sum(neighbors, 0))

    # masked recording
    spikes_rec = np.ones(ts.shape)
    for i in range(0, C):
        idxCrossing = np.where(ts[:, i] < -th)[0]
        idxCrossing = idxCrossing[np.logical_and(
            idxCrossing >= (R+1), idxCrossing <= (T-R-1))]
        spike_time = idxCrossing[np.logical_and(
            ts[idxCrossing, i] <= ts[idxCrossing-1, i],
            ts[idxCrossing, i] <= ts[idxCrossing+1, i])]

        # the portion of recording where spikes present is set to nan
        for j in np.arange(-spike_size, spike_size+1):
            spikes_rec[spike_time + j, i] = 0

    # get covariance matrix
    blanked_rec = ts*spikes_rec
    M = np.matmul(blanked_rec.transpose(), blanked_rec) / \
        np.matmul(spikes_rec.transpose(), spikes_rec)

    # since ts is standardized recording, covaraince = correlation
    invhalf_var = np.diag(np.power(np.diag(M), -0.5))
    M = np.matmul(np.matmul(invhalf_var, M), invhalf_var)

    # get localized whitening filter
    Q = np.zeros((nneigh, nneigh, C))
    for c in range(0, C):
        ch_idx, _ = order_channels_by_distance(c,
                                               np.where(neighbors[c])[0],
                                               geom)
        nneigh_c = ch_idx.shape[0]

        V, D, _ = np.linalg.svd(M[ch_idx, :][:, ch_idx])
        eps = 1e-6
        Epsilon = np.diag(1/np.power((D + eps), 0.5))
        Q_small = np.matmul(np.matmul(V, Epsilon), V.transpose())
        Q[:nneigh_c][:, :nneigh_c, c] = Q_small

    return Q


def score(score, channel_index, Q):
    """?

    Parameters
    ----------
    ?

    Returns
    -------
    ?
    """
    n_data, n_features, n_neigh = score.shape
    n_channels = Q.shape[2]

    whitened_score = np.zeros(score.shape)
    for c in range(n_channels):
        idx = channel_index == c
        whitened_score_c = np.matmul(
            np.reshape(score[idx], [-1, n_neigh]), Q[:, :, c])
        whitened_score[idx] = np.reshape(whitened_score_c,
                                         [-1, n_features, n_neigh])

    return whitened_score
