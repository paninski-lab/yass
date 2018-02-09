import numpy as np
from scipy.stats import chi2
from sklearn.cluster import KMeans


def coreset(scores, channel_index, coreset_k, coreset_th):
    """
    """
    n_channels = len(scores)
    groups = [None]*n_channels

    for channel in range(n_channels):

        neigh_channels = channel_index[channel]
        scores_channel = scores[
            channel][:, :,neigh_channels < n_channels]

        if scores_channel.shape[0] > 0:
            score_temp = np.reshape(scores_channel,
                                    [scores_channel.shape[0], -1])
            th = 1.5*np.sqrt(chi2.ppf(coreset_th, 1) *
                             score_temp.shape[1])
            
            group = coreset_alg(score_temp, 
                                th, coreset_k).astype('int32')-1
            groups[channel] = group

        else:
            groups[channel] = np.zeros(0, 'int32')

    return groups


def coreset_alg(data, th, K):
    """
    """
    if data.shape[0] > K:
        result = KMeans(n_clusters=K, init='k-means++',
                        n_init=1, max_iter=10).fit(data)
        distances = result.transform(data)
        labels = result.labels_
        label_new = np.zeros(data.shape[0])
        for k in range(K):
            idx = labels == k
            if np.max(distances[idx, k]) > th:
                label_temp = coreset_alg(data[idx], th, K)
                label_new[idx] = label_temp + np.max(label_new)
            else:
                label_new[idx] = 1 + np.max(label_new)
    else:
        label_new = np.array(range(1, data.shape[0]+1))

    return label_new