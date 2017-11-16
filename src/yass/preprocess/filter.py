"""
Filtering functions
"""

import numpy as np
from scipy.signal import butter, lfilter

from ..geometry import order_channels_by_distance, n_steps_neigh_channels


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


def whitening_matrix(ts, neighbors, spike_size):
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
    #timeRange = np.arange(0, T)
    # masked recording
    spikes_rec = np.ones(ts.shape)
    for i in range(0, C):
        #idxCrossing = timeRange[ts[:, i] < -th[i]]
        idxCrossing = np.where(ts[:, i] < -th)[0]
        idxCrossing = idxCrossing[np.logical_and(
            idxCrossing >= (R+1), idxCrossing <= (T-R-1))]
        spike_time = idxCrossing[np.logical_and(ts[idxCrossing, i] <= ts[idxCrossing-1, i],
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
        #Q_small       = np.matmul(V, np.matmul(Epsilon, V.transpose()))
        Q_small = np.matmul(np.matmul(V, Epsilon), V.transpose())
        Q[c, ch_idx] = Q_small[ch_idx == c, :]
    return Q


def whitening(ts, Q):
    return np.matmul(ts, Q.transpose())


def whiten(ts, neighbors, spike_size):
    Q = whitening_matrix(ts, neighbors, spike_size)
    return whitening(ts, Q)



def localized_whitening_matrix(ts, neighbors, geom, spike_size):
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
    nneigh = np.max(np.sum(neighbors, 0))
        
    # masked recording
    spikes_rec = np.ones(ts.shape)
    for i in range(0, C):
        #idxCrossing = timeRange[ts[:, i] < -th[i]]
        idxCrossing = np.where(ts[:, i] < -th)[0]
        idxCrossing = idxCrossing[np.logical_and(
            idxCrossing >= (R+1), idxCrossing <= (T-R-1))]
        spike_time = idxCrossing[np.logical_and(ts[idxCrossing, i] <= ts[idxCrossing-1, i],
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
        Q[:nneigh_c][:,:nneigh_c,c] = Q_small
    return Q

def whitening_score(score, channel_index, Q):
    n_data, n_features, n_neigh = score.shape
    n_channels = Q.shape[2]
    
    whitened_score = np.zeros(score.shape)
    for c in range(n_channels):
        idx = channel_index == c
        whitened_score_c = np.matmul(
            np.reshape(score[idx], [-1, n_neigh]), Q[:,:,c])
        whitened_score[idx] = np.reshape(whitened_score_c,
                                         [-1, n_features, n_neigh])
        
    return whitened_score