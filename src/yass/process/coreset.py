"""
Clustering with coresets
"""

import numpy as np
from scipy.stats import chi2
from sklearn.cluster import KMeans


# TODO: documentation
# TODO: comment code, it's not clear what it does
# FIXME: remove do_coreset
def coreset(score, coreset_k, coreset_th, do_coreset):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    n_data, n_features, n_channels = score.shape

    if do_coreset and (n_data > 0):
        score_temp = score.reshape([n_data, -1])
        th = 1.5*np.sqrt(chi2.ppf(coreset_th, 1) * score_temp.shape[1])
        group = coreset_alg(score_temp, th, coreset_k).astype('int32')-1
    else:
        group = np.arange(n_data, 'int32')

    return group


# TODO: documentation
# TODO: comment code, it's not clear what it does
def coreset_alg(data, th, K):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    if data.shape[0] > K:
        result = KMeans(n_clusters=K, init='k-means++',
                        n_init=1, max_iter=10).fit(data)
        distances = result.transform(data)
        labels = result.labels_
        label_new = np.zeros(data.shape[0])

        for k in range(K):
            idx = labels == k

            if np.sum(idx) > 0:

                if np.max(distances[idx, k]) > th:
                    label_temp = coreset_alg(data[idx], th, K)
                    label_new[idx] = label_temp + np.max(label_new)
                else:
                    label_new[idx] = 1 + np.max(label_new)
    else:
        label_new = np.array(range(1, data.shape[0]+1))

    return label_new
