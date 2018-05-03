import numpy as np
from scipy.stats import chi2
from sklearn.cluster import KMeans


def coreset(scores, spike_index, coreset_k, coreset_th):
    """
    Coreset based on hierarchical K-means

    Parameters
    ----------
    scores: list np.array(n_data, n_features, n_channels)

    spike_index: np.array(n_data, 2)

    coreset_k: int
        number of clusters for running K-means is determined by
        th = 1.5*np.sqrt(chi2.ppf(coreset_th, 1) *
                             score_temp.shape[1])

    coreset_th: float
       maximum distance allowed within each cluster

    Returns
    -------
    groups: list (n_channels)
        coreset represented as group id.
        groups[c] is the group id of spikes in scores[c]
    """

    # initialize list
    n_channels = np.max(spike_index[:, 1]) + 1
    groups = [None]*n_channels

    for channel in range(n_channels):

        idx_data = np.where(spike_index[:, 1] == channel)[0]
        scores_channel = scores[idx_data]

        # get data relevant to this channel
        n_data, n_features, n_neigh = scores_channel.shape
        # exclude empty channels
        valid_channel = np.sum(np.abs(scores_channel),
                               axis=(0, 1)) > 0
        scores_channel = scores_channel[:, :, valid_channel]

        if n_data > 0:
            score_temp = np.reshape(scores_channel, [n_data, -1])
            # calculate threshold
            th = 1.5*np.sqrt(chi2.ppf(coreset_th, 1) *
                             score_temp.shape[1])

            # run hierarchical K-means
            group = coreset_alg(score_temp,
                                th, coreset_k).astype('int32') - 1
            groups[channel] = group

        else:
            groups[channel] = np.zeros(0, 'int32')

    return groups


def coreset_alg(data, th, K):
    """
    run hierarchical K-means

    Parameters
    ----------
    data: np.array (n_data, n_dimensions)
        Data to be clustered

    th: int
        threshold to be used

    K: int
       number of clusters for each K-means

    Returns
    -------
    label_new: np.array (n_data)
       cluster id
    """
    if data.shape[0] > K:
        # run K-means
        result = KMeans(n_clusters=K, init='k-means++',
                        n_init=1, max_iter=10).fit(data)
        # calculate distance to the center
        distances = result.transform(data)

        # get label
        labels = result.labels_
        label_new = np.zeros(data.shape[0])
        for k in range(K):
            idx = labels == k
            # if distance to the center is bigger than th,
            # run again. otherwise, just add it
            if np.max(distances[idx, k]) > th:
                label_temp = coreset_alg(data[idx], th, K)
                label_new[idx] = label_temp + np.max(label_new)
            else:
                label_new[idx] = 1 + np.max(label_new)
    else:
        label_new = np.array(range(1, data.shape[0]+1))

    return label_new
