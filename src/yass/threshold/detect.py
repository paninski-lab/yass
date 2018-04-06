"""
Functions for threshold detection
"""
import os
import logging

import numpy as np

from yass.batch import BatchProcessor
from yass.util import check_for_files, LoadFile, save_numpy_object
from yass.geometry import n_steps_neigh_channels

# FIXME: seems like the detector is throwing slightly different results
# when n batch > 1
logger = logging.getLogger(__name__)


@check_for_files(filenames=[LoadFile('spike_index_clear_filename')],
                 mode='extract', relative_to='output_path')
def threshold(path_to_data, dtype, n_channels, data_order,
              max_memory, neighbors, spike_size,
              minimum_half_waveform_size, threshold, output_path=None,
              spike_index_clear_filename='spike_index_clear.npy',
              if_file_exists='skip'):
    """Threshold spike detection in batches

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

    max_memory:
        Max memory to use in each batch (e.g. 100MB, 1GB)

    neighbors_matrix: numpy.ndarray (n_channels, n_channels)
        Boolean numpy 2-D array where a i, j entry is True if i is considered
        neighbor of j

    spike_size: int
        Spike size

    minimum_half_waveform_size: int
        This is used to remove spikes that are either at the beginning or end
        of the recordings and whose location does not allow to draw a
        wavefor of size at least 2 * minimum_half_waveform_size + 1

    threshold: float
        Threshold used on amplitude for detection

    output_path: str, optional
        Directory to save spike indexes, if None, results won't be stored, but
        only returned by the function

    spike_index_clear_filename: str, optional
        Filename for spike_index_clear, it is used as the filename for the
        file (relative to output_path), if None, results won't be saved, only
        returned

    if_file_exists:
        What to do if there is already a file in save_spike_index_clear
        path. One of 'overwrite', 'abort', 'skip'. If 'overwrite' it replaces
        he file if it exists, if 'abort' if raise a ValueError exception if
        the file exists, if 'skip' it skips the operation if
        save_spike_index_clear and save_spike_index_collision the file exist
        and loads them from disk, if any of the files is missing they are
        computed

    Returns
    -------
    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    spike_index_collision: numpy.ndarray (0, 2)
        Empty array, collision is not implemented in the threshold detector
    """
    # instatiate batch processor
    bp = BatchProcessor(path_to_data, dtype, n_channels, data_order,
                        max_memory, buffer_size=spike_size)

    # run threshold detector
    spikes = bp.multi_channel_apply(_threshold,
                                    mode='memory',
                                    cleanup_function=fix_indexes,
                                    neighbors=neighbors,
                                    spike_size=spike_size,
                                    threshold=threshold)

    # no collision detection implemented, all spikes are marked as clear
    spike_index_clear = np.vstack(spikes)

    # remove spikes whose location won't let us draw a complete waveform
    logger.info('Removing clear indexes outside the allowed range to '
                'draw a complete waveform...')
    spike_index_clear, _ = (remove_incomplete_waveforms(
                            spike_index_clear,
                            minimum_half_waveform_size,
                            bp.reader.observations))

    if output_path and spike_index_clear_filename:
        path = os.path.join(output_path, spike_index_clear_filename)
        save_numpy_object(spike_index_clear, path,
                          if_file_exists=if_file_exists,
                          name='Spike index clear')

    return spike_index_clear


def _threshold(rec, neighbors, spike_size, threshold):
    """Run Threshold-based spike detection

    Parameters
    ----------
    rec: np.ndarray (n_observations, n_channels)
        numpy 2-D array with the recordings, first dimension must be
        n_observations and second n_channels

    neighbors: np.ndarray (n_channels, n_channels)
        Boolean numpy 2-D array where a i, j entry is True if i is considered
        neighbor of j

    spike_size: int
        Spike size

    threshold: float
        Threshold used on amplitude for detection

    Notes
    -----
    any values below -std_factor is considered as a spike
    and its location is saved and returned

    Returns
    -------
    index: np.ndarray (number of spikes, 2)
        First column is spike time, second column is main channel (the channel
        where spike has the biggest amplitude)
    """
    T, C = rec.shape
    R = spike_size
    th = threshold
    neigh_channels_big = n_steps_neigh_channels(neighbors, steps=2)

    # FIXME: is this a safe thing to do?
    index = np.zeros((1000000, 2), 'int32')
    count = 0

    for c in range(C):
        # For each channel, mark down location where it crosses the threshold
        idx = np.logical_and(rec[:, c] < -th, np.r_[True, rec[1:, c]
                             < rec[:-1, c]] & np.r_[rec[:-1, c]
                             < rec[1:, c], True])
        nc = np.sum(idx)

        if nc > 0:
            # location where it crosses the threshold
            spt_c = np.where(idx)[0]

            # remove an index if it is too close to the edge of the recording
            spt_c = spt_c[np.logical_and(spt_c > 2*R, spt_c < T-2*R)]
            nc = spt_c.shape[0]

            # get neighboring channels
            ch_idx = np.where(neigh_channels_big[c])[0]
            c_main = np.where(ch_idx == c)[0]

            # look at temporal spatial window around the spike location
            # if the spike being looked at has the biggest amplitude than
            # it spatial and temporal window, keep it.
            # Otherwise, disregard it
            idx_keep = np.zeros(nc, 'bool')
            for j in range(nc):
                # get waveforms
                wf_temp = rec[spt_c[j]+np.arange(-2*R, 2*R+1)][:, ch_idx]

                # location with the biggest amplitude: (t_min, c_min)
                c_min = np.argmin(np.amin(wf_temp, axis=0))
                t_min = np.argmin(wf_temp[:, c_min])
                if t_min == 2*R and c_min == c_main:
                    idx_keep[j] = 1

            nc = np.sum(idx_keep)
            index[count:(count+nc), 0] = spt_c[idx_keep]
            index[count:(count+nc), 1] = c
            count += nc

    return index[:count]


def fix_indexes(spikes, idx_local, idx, buffer_size):
    """Fixes indexes from detected spikes in batches

    Parameters
    ----------
    spikes: numpy.ndarray
        A 2D array of detected spikes as returned from detect.threshold
    idx_local: slice
        A slice object indicating the indices for the data (excluding buffer)
    idx: slice
        A slice object indicating the absolute location of the data
    buffer_size: int
        Buffer size (unused, but required to be compatible with the batch
        processor)
    """
    # remove spikes detected in the buffer area
    times = spikes[:, 0]

    data_start = idx_local[0].start
    data_end = idx_local[0].stop

    not_in_buffer = spikes[np.logical_and(times >= data_start,
                                          times <= data_end)]

    # offset spikes depending on the absolute location
    offset = idx[0].start
    not_in_buffer[:, 0] = not_in_buffer[:, 0] + offset
    return not_in_buffer


def remove_incomplete_waveforms(spike_index, spike_size, recordings_length):
    """

    Parameters
    ----------
    spikes: numpy.ndarray
        A 2D array of detected spikes as returned from detect.threshold

    Returns
    -------
    numpy.ndarray
        A new 2D array with some spikes removed. If the spike index is in a
        position (beginning or end of the recordings) where it is not possible
        to draw a complete waveform, it will be removed
    numpy.ndarray
        A boolean 1D array with True entries meaning that the index is within
        the valid range
    """
    max_index = recordings_length - 1 - spike_size
    min_index = spike_size
    include = np.logical_and(spike_index[:, 0] <= max_index,
                             spike_index[:, 0] >= min_index)
    return spike_index[include], include
