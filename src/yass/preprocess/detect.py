"""
Functions for detecting spikes
"""
import numpy as np

from ..geometry import n_steps_neigh_channels


def threshold_detection(rec, neighbors, spike_size, std_factor):
    """Threshold-based spike detection

    Parameters
    ----------
    rec
    neighbors
    spike_size
    std_factor
    n_steps_neigh_channels


    Returns
    -------
    index: matrix [number of spikes, 2]
        First column is spike time, second column is main channel (the channel
        where spike has the biggest amplitude), the third column is 1 if the
        spike has a clear spike and 0 if not.
    """
    T, C = rec.shape
    R = spike_size
    th = std_factor
    neighChannels_big = n_steps_neigh_channels(neighbors, steps=2)

    index = np.zeros((1000000, 2), 'int32')
    count = 0

    for c in range(C):
        idx = np.logical_and(rec[:, c] < -th, np.r_[True, rec[1:, c]
                                                    < rec[:-1, c]] & np.r_[rec[:-1, c] < rec[1:, c], True])
        nc = np.sum(idx)
        if nc > 0:
            spt_c = np.where(idx)[0]
            spt_c = spt_c[np.logical_and(spt_c > 2*R, spt_c < T-2*R)]
            nc = spt_c.shape[0]
            ch_idx = np.where(neighChannels_big[c])[0]
            c_main = np.where(ch_idx == c)[0]
            idx_keep = np.zeros(nc, 'bool')
            for j in range(nc):
                wf_temp = rec[spt_c[j]+np.arange(-2*R, 2*R+1)][:, ch_idx]
                c_min = np.argmin(np.amin(wf_temp, axis=0))
                t_min = np.argmin(wf_temp[:, c_min])
                if t_min == 2*R and c_min == c_main:
                    idx_keep[j] = 1

            nc = np.sum(idx_keep)
            index[count:(count+nc), 0] = spt_c[idx_keep]
            index[count:(count+nc), 1] = c
            count += nc

    return index[:count]
