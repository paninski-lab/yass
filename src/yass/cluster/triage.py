import numpy as np
from scipy.spatial import cKDTree


def triage(scores, spike_index, triage_k,
           triage_percent, location_feature):
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
    n_channels = np.max(spike_index[:, 1]) + 1
    th = (1 - triage_percent)*100

    idx_triage = np.zeros(scores.shape[0], 'bool')
    for channel in range(n_channels):
        idx_data = np.where(spike_index[:, 1] == channel)[0]
        scores_channel = scores[idx_data]
        nc = scores_channel.shape[0]

        if nc > triage_k + 1:
            if location_feature:
                scores_channel = scores_channel[:, :, 0]
                th = (1 - triage_percent/2)*100

                # get distance to nearest neighbors
                tree = cKDTree(scores_channel[:, :2])
                dist, ind = tree.query(scores_channel[:, :2], k=triage_k + 1)
                dist = np.sum(dist, 1)
                # triage far ones
                idx_triage[idx_data[dist > np.percentile(dist, th)]] = 1

                # get distance to nearest neighbors
                tree = cKDTree(scores_channel[:, 2:])
                dist, ind = tree.query(scores_channel[:, 2:], k=triage_k + 1)
                dist = np.sum(dist, 1)
                # triage far ones
                idx_triage[idx_data[dist > np.percentile(dist, th)]] = 1

            else:
                n_neigh = scores_channel.shape[2]
                th = (1 - triage_percent/n_neigh)*100

                for c in range(n_neigh):
                    tree = cKDTree(scores_channel[:, :, c])
                    dist, ind = tree.query(scores_channel[:, :, c],
                                           k=triage_k + 1)
                    dist = np.sum(dist, 1)
                    # triage far ones
                    idx_triage[idx_data[dist > np.percentile(dist, th)]] = 1

    idx_triage = np.where(idx_triage)[0]
    scores = np.delete(scores, idx_triage, 0)
    spike_index = np.delete(spike_index, idx_triage, 0)

    return scores, spike_index
