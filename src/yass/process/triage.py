"""
Triage spikes in clean or collision
"""

import numpy as np
from scipy.spatial import cKDTree


# FIXME: why is there a do_triage parameter on this function?
# TODO: documentation
# TODO: comment code, it's not clear what it does
def triage(score, channel_index, triage_k, triage_percent, do_triage):
    """[?]

    Parameters
    ----------
    ?

    Note
    ----
    This object mutates score and clr_idx
    """

    K = triage_k + 1
    th = (1 - triage_percent)*100
    nmax = 10000

    n_data, n_features, n_channels = score.shape
    index_keep_bool = np.zeros(n_data, 'bool')
    if n_data > nmax:
        index_keep = np.random.permutation(n_data)[:nmax]
        score = score[index_keep]
    else:
        index_keep = np.arange(n_data)

    if do_triage and (n_data > K):
        score_temp = score[:, :, channel_index]
        tree = cKDTree(score_temp)
        dist, ind = tree.query(score_temp, k=K)
        dist = np.sum(dist, 1)
        index_keep = index_keep[dist < np.percentile(dist, th)]

    index_keep_bool[index_keep] = 1

    return index_keep_bool
