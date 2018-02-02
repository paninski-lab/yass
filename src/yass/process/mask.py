from scipy.stats import chi2
import numpy as np


# TODO: documentation
# TODO: comment code, it's not clear what it does
def getmask(score, group, mask_th, n_features):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    th = 1.5*(chi2.ppf(mask_th, 1)*n_features)
    n_data, n_features, n_channels = score.shape
    if n_data > 0:
        n_group = np.max(group) + 1

        # find the average score per group
        score_group = np.zeros((n_group, n_features, n_channels))
        n_per_group = np.zeros(n_group)

        for j in range(n_data):
            score_group[group[j]] += score[j]
            n_per_group[group[j]] += 1
        for j in range(n_group):
            score_group[j] = score_group[j]/n_per_group[j]

        # find mask for each averaged score
        maskTemp = np.minimum(np.maximum(
            ((np.sum(np.square(score_group), axis=1)
             - np.min(th))/(np.max(th)-np.min(th))), 0), 1)

        # match the mask per group to each data
        mask = np.zeros((n_data, n_channels))
        for j in range(n_data):
            mask[j] = maskTemp[group[j]]

    return mask
