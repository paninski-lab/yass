import numpy as np
from scipy.spatial import cKDTree


def triage(scores, spike_index, triage_k, triage_percent):
    """
    Triage based on KNN distance.
    It removes triage_percent*100% of data

    Parameters
    ----------
    scores: list (n_channels)
        A list such that scores[c] contains all scores whose main
        channel is c

    spike_index: list (n_channels)
        A list such that spike_index[c] cointains all spike times
        whose channel is c

    triage_k: int
       number of neighbors to consider

    triage_percent: float
       percentage of data to be triaged.
       It is a number between 0 and 1.

    Returns
    -------
    scores: list (n_channels)
        scores after triage

    spike_index: list (n_channels)
        spike_index after traige
    """
    # relevant info
    n_channels = len(scores)
    th = (1 - triage_percent)*100

    for channel in range(n_channels):
        scores_channel = scores[channel][:, :, 0]
        nc = scores_channel.shape[0]

        if nc > triage_k + 1:
            # get distance to nearest neighbors
            tree = cKDTree(scores_channel)
            dist, ind = tree.query(scores_channel, k=triage_k + 1)
            dist = np.sum(dist, 1)

            # triage far ones
            idx_keep = dist < np.percentile(dist, th)
            scores[channel] = scores[channel][idx_keep]
            spike_index[channel] = spike_index[channel][idx_keep]

    return scores, spike_index
