"""
Whitening functions
"""
try:
    from pathlib2 import Path
except Exception:
    from pathlib import Path

import logging

import numpy as np

from yass.batch import BatchProcessor
from yass.util import save_numpy_object, check_for_files, LoadFile


@check_for_files(filenames=[LoadFile('output_filename')],
                 mode='extract', relative_to='output_path')
def matrix(path_to_data, dtype, n_channels, data_order,
           channel_index, spike_size, max_memory, output_path,
           output_filename='whitening.npy',
           if_file_exists='skip'):
    """Compute whitening filter using the first batch of the data

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

    channel_index: np.array
        A matrix of size [n_channels, n_nieghbors], showing neighboring channel
        information

    spike_size: int
        Spike size

    max_memory: str
        Max memory to use in each batch (e.g. 100MB, 1GB)

    output_path: str
        Where to store the whitenint gilter

    output_filename: str, optional
        Filename for the output data, defaults to whitening.npy

    if_file_exists: str, optional
        One of 'overwrite', 'abort', 'skip'. If 'overwrite' it replaces the
        whitening filter if it exists, if 'abort' if raise a ValueError
        exception if the file exists, if 'skip' if skips the operation if the
        file exists

    Returns
    -------
    standarized_path: str
        Path to standarized recordings

    standarized_params: dict
        A dictionary with the parameters for the standarized recordings
        (dtype, n_channels, data_order)
    """
    logger = logging.getLogger(__name__)

    # compute Q (using the first batchfor whitening
    logger.info('Computing whitening matrix...')

    bp = BatchProcessor(path_to_data, dtype, n_channels, data_order,
                        max_memory)

    batches = bp.multi_channel()
    first_batch = next(batches)
    whiten_filter = _matrix(first_batch, channel_index, spike_size)

    path_to_whitening_matrix = Path(output_path, output_filename)
    save_numpy_object(whiten_filter, path_to_whitening_matrix,
                      if_file_exists='overwrite',
                      name='whitening filter')

    return whiten_filter


def _matrix(recording, channel_index, spike_size):
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
