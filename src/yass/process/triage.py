import numpy as np
from scipy.spatial import cKDTree


# FIXME: dotriage should not be here
def triage(scores, spike_index, triage_k, triage_percent):
    """

    Note
    ----
    This object mutates score and clr_idx
    """
    
    n_channels = len(scores)
    
    K = triage_k + 1
    th = (1 - triage_percent)*100

    for channel in range(n_channels):
        scores_channel = scores[c][:,:,0]
        nc = scores_channel.shape[0]

        if nc > K:
            tree = cKDTree(scores_channel)
            dist, ind = tree.query(scores_channel, k=K)
            idx_keep = dist < np.percentile(np.sum(dist, 1), th)
            
            scores[c] = scores[c][idx_keep]
            spike_index[c] = spike_index[c][idx_keep]

    return scores, spike_index