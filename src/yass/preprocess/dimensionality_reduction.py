"""
Functions for dimensionality reduction
"""
import logging

import numpy as np

from ..geometry import ordered_neighbors

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


def score(waveforms, rot, main_channels=None, neighbors=None, geom=None):
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
    if waveforms.ndim != 3:
        raise ValueError('waveforms must have dimension 3 (has {})'.format(
            waveforms.ndim))

    n_waveforms, n_temporal_features, n_channels = waveforms.shape

    if rot.ndim == 2:
        # neural net case
        n_temporal_features_, n_reduced_features = rot.shape

        if n_temporal_features != n_temporal_features_:
            raise ValueError('n_temporal_features does not match between '
                             'waveforms ({}) and the rotation matrix ({})'
                             .format(n_temporal_features,
                                     n_temporal_features_))

        reduced = np.matmul(rot.T, waveforms)

    elif rot.ndim == 3:
        # pca case
        n_temporal_features_, n_reduced_features, n_channels = rot.shape

        if n_temporal_features != n_temporal_features_:
            raise ValueError('n_temporal_features does not match between '
                             'waveforms ({}) and the rotation matrix ({})'
                             .format(n_temporal_features,
                                     n_temporal_features_))

        reduced = np.matmul(rot.T, waveforms.T).T

    else:
        raise ValueError('rot must have 2 or 3 dimensions (has {})'.format(
            rot.ndim))

    # if passed information about neighbors, get scores only for them instead
    # of all channels
    if (main_channels is not None and neighbors is not None
            and geom is not None):
        # for every spike, get the score only for the neighboring channels
        ord_neighbors, channel_features = ordered_neighbors(geom, neighbors)
        score_neigh = np.zeros((n_waveforms, n_reduced_features,
                                channel_features))

        for i in range(n_waveforms):
            # get the ordered neighbors for the main channel
            current_neigh = ord_neighbors[main_channels[i]]
            # assign the scores for those channels to the matrix
            score_neigh[i, :, :len(current_neigh)] = reduced[i][:,
                                                                current_neigh]

        return score_neigh

    else:
        return reduced


def main_channel_scores(waveforms, rot, spike_index, CONFIG):
    """Returns PCA scores for the main channel only

    Parameters
    ----------
    waveforms: numpy.ndarray
    rot: numpy.ndarray (window_size,n_features, n_channels)
        PCA rotation matrix
    spike_index: np.ndarray (number of spikes, 2)
        Spike indexes as returned from the threshold detector
    """
    if CONFIG.spikes.detection == 'threshold':
        spikes, _, n_channels = waveforms.shape
        _, n_features, _ = rot.shape

        score = np.zeros([spikes, n_features])
        main_channel = spike_index[:, 1]

        for i in range(spikes):
            score[i, :] = np.squeeze(
                np.matmul(waveforms[i, :, main_channel[i]][np.newaxis],
                          rot[:, :, main_channel[i]]))

    else:
        spikes, _, n_channels = waveforms.shape
        _, n_features = rot.shape

        score = np.zeros([spikes, n_features])
        main_channel = spike_index[:, 1]

        for i in range(spikes):
            score[i, :] = np.squeeze(
                np.matmul(waveforms[i, :, main_channel[i]][np.newaxis], rot))

    return score


def denoise(waveforms, rot, CONFIG):
    """Denoise waveforms by projecting into PCA space and back

    Parameters
    ----------
    Waveforms: numpy.ndarray
    rot: numpy.ndarray (window_size, n_features, n_channels)
        PCA Rotation matrix
    """
    if CONFIG.spikes.detection == 'threshold':
        rot_ = np.transpose(rot)
        rot2_ = np.transpose(rot_, [0, 2, 1])

        denoising_rot = np.matmul(rot2_, rot_)
        sp = np.transpose(waveforms, [0, 2, 1])

        denoised_waveforms = np.transpose(
            np.squeeze(np.matmul(sp[:, :, np.newaxis], denoising_rot), axis=2),
            [0, 2, 1])
    else:
        rot_ = np.transpose(rot)

        denoising_rot = np.matmul(rot, rot_)
        sp = np.transpose(waveforms)

        denoised_waveforms = np.transpose(np.matmul(denoising_rot, sp))

    return denoised_waveforms
