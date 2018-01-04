"""
Filtering functions

Original source:
https://github.com/hooshmandshr/yass_visualization/blob/master/src/stability/filtering.py
"""

import numpy as np
from scipy.signal import butter, lfilter

from ..geometry import n_steps_neigh_channels

# FIXME: these functions were copied from yass when this was in a separate repo
# the yass versions have been updated, we need to update the stability copied
# to make use of the new yass functions and remove this file


def butterworth(ts, low_freq, high_factor, order, sampling_freq):
    """Butterworth filter of for time series
    Parameters
    ----------
    ts: np.array
        T x C numpy array, where T is the number of time samples and
        C is the number of channels
    low_freq: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_freq: int
        Sampling frequency (Hz)
    """
    T, C = ts.shape

    low = float(low_freq)/sampling_freq * 2
    high = float(high_factor) * 2
    b, a = butter(order, [low, high], btype='band')
    fts = np.zeros((T, C))

    for i in range(C):
        fts[:, i] = lfilter(b, a, ts[:, i])

    return fts


def whitening(ts, neighbors, spike_size):
    """Spatial whitening filter for time series
    Parameters
    ----------
    ts: np.array
        T x C numpy array, where T is the number of time samples and
        C is the number of channels
    """
    # get all necessary parameters from param
    [T, C] = ts.shape
    R = spike_size*2 + 1
    th = 4
    neighChannels = n_steps_neigh_channels(neighbors, steps=2)

    chanRange = np.arange(0, C)
    # timeRange = np.arange(0, T)
    # masked recording
    spikes_rec = np.ones(ts.shape)

    for i in range(0, C):
        # idxCrossing = timeRange[ts[:, i] < -th[i]]
        idxCrossing = np.where(ts[:, i] < -th)[0]
        idxCrossing = idxCrossing[np.logical_and(
            idxCrossing >= (R+1), idxCrossing <= (T-R-1))]
        spike_time = idxCrossing[np.logical_and(ts[idxCrossing, i] <=
                                                ts[idxCrossing-1, i],
                                                ts[idxCrossing, i] <=
                                                ts[idxCrossing+1, i])]

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
        Epsilon = np.diag(1/np.power((D), 0.5))
        Q_small = np.matmul(np.matmul(V, Epsilon), V.transpose())
        Q[c, ch_idx] = Q_small[ch_idx == c, :]

    return np.matmul(ts, Q.transpose())
