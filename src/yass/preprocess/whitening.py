import numpy as np

from ..geometry import n_steps_neigh_channels


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
    spikes_rec = np.ones(ts.shape)

    for i in range(0, C):
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
        Q_small = np.matmul(np.matmul(V, Epsilon), V.transpose())
        Q[c, ch_idx] = Q_small[ch_idx == c, :]

    return Q


def whitening(ts, Q):
    return np.matmul(ts, Q.transpose())
