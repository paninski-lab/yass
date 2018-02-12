import numpy as np

def random_subsample(scores, spike_index, n_sample=1000):
    
    n_channels = len(scores)
    
    for c in range(n_channels):
        
        n_data = scores[c].shape[0]

        idx_keep = np.random.choice(n_data,
                                    np.min((n_sample, n_data)),
                                    replace=False)
        scores[c] = scores[c][idx_keep]
        spike_index[c] = spike_index[c][idx_keep]
        
    return scores, spike_index
        
        
    
    
    