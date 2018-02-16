"""
Functions for detecting spikes
"""
import numpy as np

from yass.geometry import n_steps_neigh_channels

# FIXME: seems like the detector is throwing slightly different results
# when n batch > 1


def run(rec, neighbors, spike_size, threshold):
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
    threshold: float?
        threshold used on amplitude for detection

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
    neighChannels_big = n_steps_neigh_channels(neighbors, steps=2)

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
            ch_idx = np.where(neighChannels_big[c])[0]
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
