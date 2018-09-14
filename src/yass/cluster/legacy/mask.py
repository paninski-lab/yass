from scipy.stats import chi2
import numpy as np


def getmask(scores, spike_index, groups, mask_th):
    """
    Get mask of each data

    Parameters
    ----------
    scores: list np.array(n_data, n_features, n_channels)

    spike_index: np.array(n_data, 2)

    groups: list (n_channels)
        coreset information

    mask_th: np.array (2)
       a strong and weak threshold for mask

    Returns
    -------
    masks: list (n_channels)
        mask for each data in scores
        masks[c] is the mask of spikes in scores[c]
    """

    # initialize
    n_channels = np.max(spike_index[:, 1]) + 1
    masks = [None]*n_channels

    for channel in range(n_channels):

        idx_data = np.where(spike_index[:, 1] == channel)[0]

        # get score and group for this channel
        score_channel = scores[idx_data]
        group_channel = groups[channel]

        if score_channel.shape[0] > 0:
            # get shape and threshold
            n_data, n_features, n_neigh = score_channel.shape
            th = 1.5*(chi2.ppf(mask_th, 1)*n_features)

            # number of coresets
            n_group = np.max(group_channel) + 1

            # get average score per group
            score_group = np.zeros((n_group, n_features, n_neigh))
            n_per_group = np.zeros(n_group)
            for j in range(n_data):
                score_group[group_channel[j]] += score_channel[j]
                n_per_group[group_channel[j]] += 1

            for j in range(n_group):
                score_group[j] = score_group[j]/n_per_group[j]

            # get energy (l2 norm of score)
            energy_group = np.sum(np.square(score_group), axis=1)

            # determine mask using energy
            mask_group = np.minimum(np.maximum(
                (energy_group - np.min(th))/(np.max(th)-np.min(th)), 0), 1)

            # mask is redistributed per data
            # since it was calculated per group
            mask_channel = np.zeros((n_data, n_neigh))
            for j in range(n_data):
                mask_channel[j] = mask_group[group_channel[j]]

            masks[channel] = mask_channel

        else:
            masks[channel] = np.zeros(0)

    return masks
