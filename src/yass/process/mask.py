from scipy.stats import chi2
import numpy as np


def getmask(scores, groups, mask_th):
    """
    """
    
    n_channels = len(scores)
    masks = [None]*n_channels
    
    for channel in range(n_channels):
        score_channel = scores[channel]
        group_channel = groups[channel]
        
        if score_channel.shape[0] > 0:
            n_data, n_features, n_neigh = score_channel.shape
            th = 1.5*(chi2.ppf(mask_th, 1)*n_features)
            
            n_group = np.max(group_channel) + 1

            score_group = np.zeros((n_group, n_features, n_neigh))
            n_per_group = np.zeros(n_group)
            for j in range(n_data):
                score_group[group_channel[j]] += score_channel[j]
                n_per_group[group_channel[j]] += 1
                
            for j in range(n_group):
                score_group[j] = score_group[j]/n_per_group[j]

            energy_group = np.sum(np.square(score_group), axis=1)
            
            mask_group = np.minimum(np.maximum(
                (energy_group - np.min(th))/(np.max(th)-np.min(th)), 0), 1)
            
            mask_channel = np.zeros((n_data, n_neigh))
            for j in range(n_data):
                mask_channel[j] = mask_group[group_channel[j]]
                
            masks[channel] = mask_channel

        else:
            masks[channel] = np.zeros(0)

    return masks