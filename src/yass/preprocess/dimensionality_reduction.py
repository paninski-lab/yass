"""
Functions for dimensionality reduction
"""
import logging

import numpy as np

from yass.geometry import ordered_neighbors

# TODO: improve documentation
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
    window_idx = range(-spike_size, spike_size + 1)
    window_size = len(window_idx)

    pca_suff_stat = np.zeros((window_size, window_size, n_channels))
    spikes_per_channel = np.zeros(n_channels, 'int32')

    # iterate over every channel
    for c in range(n_channels):
        # get spikes times for the current channel
        channel_spike_times = spike_index[spike_index[:, MAIN_CHANNEL] == c,
                                          SPIKE_TIME]
        channel_spike_times = channel_spike_times[np.logical_and(
            (channel_spike_times > spike_size),
            (channel_spike_times < n_obs - spike_size - 1))]

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
    rot_all = v[:,
                np.argsort(w)[window_size:(window_size - n_features - 1):-1]]

    for c in range(n_channels):
        if spikes_per_channel[c] <= window_size:
            if np.sum(spikes_per_channel[neighbors[c, :]]) <= window_size:
                rot[:, :, c] = rot_all
            else:
                w, v = np.linalg.eig(np.sum(ss[:, :, neighbors[c, :]], 2))
                rot[:, :, c] = v[:,
                                 np.argsort(w)[window_size:(
                                     window_size - n_features - 1):-1]]
        else:
            w, v = np.linalg.eig(ss[:, :, c])
            rot[:, :, c] = v[:,
                             np.argsort(w)[window_size:(
                                 window_size - n_features - 1):-1]]

    return rot


def score(recording, rot, channel_index, spike_index):
    """
    Reduce waveform dimensionality using a rotation matrix. Optionally
    return scores only for neighboring channels instead of all channels

    Parameters
    ----------
    waveforms: numpy.ndarray (n_waveforms, n_temporal_features, n_channels)
        Waveforms to score

    rot: numpy.ndarray
        Rotation matrix. Array with dimensions (n_temporal_features,
        n_features, n_channels) for PCA matrix or (n_temporal_features,
        n_features) for autoencoder matrix

    main_channels: numpy.ndarray (n_waveforms), optional
        Main channel (biggest amplitude) for each waveform

    neighbors: numpy.ndarray (n_channels, n_channels), optional
        Neighbors matrix

    geom: numpy.ndarray (n_channels, 2), optional
        Channels location matrix

    Returns
    -------
    (n_waveforms, n_reduced_features, n_channels/n_neighboring_channels)
        Scores for every waveform, second dimension in the array is reduced
        from n_temporal_features to n_reduced_features, third dimension
        is n_channels if no information about the main channel and geometry is
        passed (main_channels, neighbors, geom), otherwise this information
        is used to only return the scores for the neighboring channels for
        the main channel in each waveform so the last dimension is
        n_neighboring_channels
    """

    n_observations, n_channels = recording.shape
    n_data = spike_index.shape[0]
    n_neigh = channel_index.shape[1]

    if rot.ndim == 2:
        # neural net case
        n_temporal_features, n_reduced_features = rot.shape
        
        rot = np.tile(rot[:, :, np.newaxis], [1, 1, n_channels])

    elif rot.ndim == 3:
        # pca case
        n_temporal_features, n_features, n_channels_ = rot.shape

        if n_channels != n_channels_:
            raise ValueError('n_channels does not match between '
                             'recording ({}) and the rotation matrix ({})'
                             .format(n_channels,
                                     n_channels_))
            
    else:
        raise ValueError('rot must have 2 or 3 dimensions (has {})'.format(
            rot.ndim))
        
    if n_temporal_features % 2 != 1:
        raise ValueError('waveform length needs to be an odd number (has {})'.format(
            n_temporal_features))        
    R = int((n_temporal_features-1)/2)
    
    rot = np.transpose(rot, [2, 1, 0])
    scores = np.zeros((n_data, n_features, n_neigh))        
    for channel in range(n_channels):
        
        ch_idx = channel_index[channel][
            channel_index[channel] < n_channels]
        
        idx_c = spike_index[:, 1] == channel
        spt_c = spike_index[idx_c, 0] 
        waveforms = np.zeros((spt_c.shape[0], ch_idx.shape[0], n_temporal_features))
        
        for j in range(spt_c.shape[0]):
            waveforms[j] = recording[spt_c[j]-R:spt_c[j]+R+1, ch_idx].T
            
        scores[idx_c, :, :ch_idx.shape[0]] = np.transpose(
            np.matmul(np.expand_dims(rot[ch_idx], 0), 
                      np.expand_dims(waveforms, -1))[:, :, :, 0], 
            [0, 2, 1])
        
    return scores