"""
Whitening functions
"""

import numpy as np


def matrix(recording, channel_index, spike_size):
    """
    Spatial whitening filter for time series
    For every channel, spatial whitening filter is calculated along
    with its neighboring channels.

    Parameters
    ----------
    recording: np.array (n_observations, n_channels)
        n_observations is the number of time samples and
        n_channels is the number of channels

    channel_index: np.array (n_channels, n_neigh)
        Each row indexes its neighboring channels.
        For example, channel_index[c] is the index of
        neighboring channels (including itself)
        If any value is equal to n_channels, it is nothing but
        a space holder in a case that a channel has less than
        n_neigh neighboring channels

    spike_size: int
        half of waveform temporal spike size in number of time bins.


    Returns
    -------
    whiten_filter: numpy.ndarray (n_channels, n_neigh, n_neigh)
        whitening matrix such that whiten_filter[c] is the whitening
        filter of channel c and its neighboring channel determined from
        channel_index.
    """
    # get all necessary parameters
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
    """
    Whiten scores using whitening filter

    Parameters
    ----------
    scores: np.array (n_data, n_features, n_neigh)
        n_data is the number of spikes
        n_feature is the number features
        n_neigh is the number of neighboring channels considered

    main_channel: np.array (n_data,)
        The main channel information for each spike

    whilten_filter: np.array (n_channels, n_neigh, n_neigh)
        whitening filter as described above

    Returns
    -------
    whiten_scores: np.array (n_data, n_features, n_neigh)
        scores whitened after applying whitening filter
    """
    # get necessary parameters
    n_data, n_features, n_neigh = scores.shape
    n_channels = whiten_filter.shape[0]

    # apply whitening filter
    whitened_scores = np.zeros(scores.shape)
    for c in range(n_channels):
        # index of spikes with main channel as c
        idx = main_channel == c
        whitened_scores_c = np.matmul(
            np.reshape(scores[idx], [-1, n_neigh]), whiten_filter[c])
        whitened_scores[idx] = np.reshape(whitened_scores_c,
                                          [-1, n_features, n_neigh])

    return whitened_scores
