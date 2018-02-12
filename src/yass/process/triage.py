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
    
    th = (1 - triage_percent)*100

    for channel in range(n_channels):
        scores_channel = scores[channel][:, :, 0]
        nc = scores_channel.shape[0]

        if nc > triage_k + 1:
            tree = cKDTree(scores_channel)
            dist, ind = tree.query(scores_channel, k=triage_k + 1)
            dist = np.sum(dist, 1)
            idx_keep = dist < np.percentile(dist, th)
            
            scores[channel] = scores[channel][idx_keep]
            spike_index[channel] = spike_index[channel][idx_keep]

    return scores, spike_index