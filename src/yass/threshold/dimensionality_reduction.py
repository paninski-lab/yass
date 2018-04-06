"""
Functions for dimensionality reduction
"""
try:
    from pathlib2 import Path
except Exception:
    from pathlib import Path

from functools import reduce
import logging

import numpy as np

from yass.batch import BatchProcessor
from yass.util import check_for_files, LoadFile, save_numpy_object

logger = logging.getLogger(__name__)


@check_for_files(filenames=[LoadFile('scores_filename'),
                            LoadFile('spike_index_clear_filename'),
                            LoadFile('rotation_matrix_filename')],
                 mode='extract', relative_to='output_path')
def pca(path_to_data, dtype, n_channels, data_order, recordings, spike_index,
        spike_size, temporal_features, neighbors_matrix, channel_index,
        max_memory, output_path=None, scores_filename='scores.npy',
        rotation_matrix_filename='rotation.npy',
        spike_index_clear_filename='spike_index_clear_pca.npy',
        if_file_exists='skip'):
    """Apply PCA in batches

    Parameters
    ----------
    path_to_data: str
        Path to recordings in binary format

    dtype: str
        Recordings dtype

    n_channels: int
        Number of channels in the recordings

    data_order: str
        Recordings order, one of ('channels', 'samples'). In a dataset with k
        observations per channel and j channels: 'channels' means first k
        contiguous observations come from channel 0, then channel 1, and so
        on. 'sample' means first j contiguous data are the first observations
        from all channels, then the second observations from all channels and
        so on

    recordings: np.ndarray (n_observations, n_channels)
        Multi-channel recordings

    spike_index: numpy.ndarray
        A 2D numpy array, first column is spike time, second column is main
        channel (the channel where spike has the biggest amplitude)

    spike_size: int
        Spike size

    temporal_features: numpy.ndarray
        Number of output features

    neighbors_matrix: numpy.ndarray (n_channels, n_channels)
        Boolean numpy 2-D array where a i, j entry is True if i is considered
        neighbor of j

    channel_index: np.array (n_channels, n_neigh)
        Each row indexes its neighboring channels.
        For example, channel_index[c] is the index of
        neighboring channels (including itself)
        If any value is equal to n_channels, it is nothing but
        a space holder in a case that a channel has less than
        n_neigh neighboring channels


    max_memory:
        Max memory to use in each batch (e.g. 100MB, 1GB)

    output_path: str, optional
        Directory to store the scores and rotation matrix, if None, previous
        results on disk are ignored, operations are computed and results
        aren't saved to disk

    scores_filename: str, optional
        File name for rotation matrix if False, does not save data

    rotation_matrix_filename: str, optional
        File name for scores if False, does not save data

    spike_index_clear_filename: str, optional
        File name for spike index clear

    if_file_exists:
        What to do if there is already a file in the rotation matrix and/or
        scores location. One of 'overwrite', 'abort', 'skip'. If 'overwrite'
        it replaces the file if it exists, if 'abort' if raise a ValueError
        exception if the file exists, if 'skip' if skips the operation if the
        file exists

    Returns
    -------
    scores: numpy.ndarray
        Numpy 3D array  of size (n_waveforms, n_reduced_features,
        n_neighboring_channels) Scores for every waveform, second dimension in
        the array is reduced from n_temporal_features to n_reduced_features,
        third dimension depends on the number of  neighboring channels

    rotation_matrix: numpy.ndarray
        3D array (window_size, n_features, n_channels)
    """

    ###########################
    # compute rotation matrix #
    ###########################

    bp = BatchProcessor(path_to_data, dtype, n_channels, data_order,
                        max_memory, buffer_size=spike_size)

    # compute PCA sufficient statistics
    logger.info('Computing PCA sufficient statistics...')
    stats = bp.multi_channel_apply(suff_stat, mode='memory',
                                   spike_index=spike_index,
                                   spike_size=spike_size)
    suff_stats = reduce(lambda x, y: np.add(x, y), [e[0] for e in stats])
    spikes_per_channel = reduce(lambda x, y: np.add(x, y),
                                [e[1] for e in stats])

    # compute PCA projection matrix
    logger.info('Computing PCA projection matrix...')
    rotation = project(suff_stats, spikes_per_channel, temporal_features,
                       neighbors_matrix)

    #####################################
    # waveform dimensionality reduction #
    #####################################

    logger.info('Reducing spikes dimensionality with PCA matrix...')
    res = bp.multi_channel_apply(score,
                                 mode='memory',
                                 pass_batch_info=True,
                                 rot=rotation,
                                 channel_index=channel_index,
                                 spike_index=spike_index)

    scores = np.concatenate([element[0] for element in res], axis=0)
    spike_index = np.concatenate([element[1] for element in res], axis=0)

    # save scores
    if output_path and scores_filename:
        path_to_score = Path(output_path) / scores_filename
        save_numpy_object(scores, path_to_score,
                          if_file_exists=if_file_exists,
                          name='scores')

    if output_path and spike_index_clear_filename:
        path_to_spike_index = Path(output_path) / spike_index_clear_filename
        save_numpy_object(spike_index, path_to_spike_index,
                          if_file_exists=if_file_exists,
                          name='Spike index PCA')

    if output_path and rotation_matrix_filename:
        path_to_rotation = Path(output_path) / rotation_matrix_filename
        save_numpy_object(rotation, path_to_rotation,
                          if_file_exists=if_file_exists,
                          name='rotation matrix')

    return scores, spike_index, rotation


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


def score(recording, idx_local, idx, rot, channel_index, spike_index):
    """
    Reduce waveform dimensionality using a rotation matrix. Optionally
    return scores only for neighboring channels instead of all channels

    Parameters
    ----------
    recordings: np.ndarray (n_observations, n_channels)
        Multi-channel recordings

    rot: numpy.ndarray
        Rotation matrix. Array with dimensions (n_temporal_features,
        n_features, n_channels) for PCA matrix or (n_temporal_features,
        n_features) for autoencoder matrix

    channel_index: np.array (n_channels, n_neigh)
        Each row indexes its neighboring channels.
        For example, channel_index[c] is the index of
        neighboring channels (including itself)
        If any value is equal to n_channels, it is nothing but
        a space holder in a case that a channel has less than
        n_neigh neighboring channels

    spike_index: np.array (n_spikes, 2)
        contains spike information, the first column is the
        spike time and the second column is the main channel

    Returns
    -------
    scores: np.array (n_spikes, n_features, n_neighboring_channels)
        Scores for every waveform, second dimension in the array is reduced
        from n_temporal_features to n_features, third dimension
        is number of neighboring channels.
    """

    data_start = idx[0].start
    data_end = idx[0].stop
    # get offset that will be applied
    offset = idx_local[0].start

    spike_time = spike_index[:, 0]
    spike_index = spike_index[np.logical_and(spike_time >= data_start,
                                             spike_time < data_end)]
    spike_index[:, 0] = spike_index[:, 0] - data_start + offset

    # obtain shape information
    n_observations, n_channels = recording.shape
    n_data = spike_index.shape[0]
    n_neigh = channel_index.shape[1]

    # if rot has two dimension, rotation matrix is used for every
    # channels, if it is three, the third dimension has to match
    # the number of channels
    if rot.ndim == 2:
        # neural net case
        n_temporal_features, n_reduced_features = rot.shape
        # copy rotation matrix to all channels
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

    # n_temporal_features has to be an odd number
    if n_temporal_features % 2 != 1:
        raise ValueError('waveform length needs to be'
                         'an odd number (has {})'.format(
                             n_temporal_features))

    R = int((n_temporal_features-1)/2)

    rot = np.transpose(rot, [2, 1, 0])
    scores = np.zeros((n_data, n_features, n_neigh))
    for channel in range(n_channels):

        # get neighboring channel information
        ch_idx = channel_index[channel][
            channel_index[channel] < n_channels]

        # get spikes whose main channel is equal to channel
        idx_c = spike_index[:, 1] == channel

        # get waveforms
        spt_c = spike_index[idx_c, 0]
        waveforms = np.zeros((spt_c.shape[0], ch_idx.shape[0],
                              n_temporal_features))
        for j in range(spt_c.shape[0]):
            waveforms[j] = recording[spt_c[j]-R:spt_c[j]+R+1, ch_idx].T

        # apply rot on wavefomrs
        scores[idx_c, :, :ch_idx.shape[0]] = np.transpose(
            np.matmul(np.expand_dims(rot[ch_idx], 0),
                      np.expand_dims(waveforms, -1))[:, :, :, 0],
            [0, 2, 1])

    spike_index[:, 0] = spike_index[:, 0] + data_start - offset

    return scores, spike_index
