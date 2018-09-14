import numpy as np


def random_subsample(scores, spike_index, n_sample):
    """
    Triage based random subsampling

    Parameters
    ----------
    scores: list (n_channels)
        A list such that scores[c] contains all scores whose main
        channel is c

    spike_index: list (n_channels)
        A list such that spike_index[c] cointains all spike times
        whose channel is c

    n_sample: int
        maximum number of samples to keep

    Returns
    -------
    scores: list (n_channels)
        scores after triage

    spike_index: list (n_channels)
        spike_index after traige
    """
    n_channels = np.max(spike_index[:, 1]) + 1

    idx_keep = np.zeros(spike_index.shape[0], 'bool')
    for channel in range(n_channels):
        idx_data = np.where(spike_index[:, 1] == channel)[0]
        n_data = idx_data.shape[0]

        if n_data > n_sample:
            idx_sample = np.random.choice(n_data,
                                          n_sample,
                                          replace=False)
            idx_keep[idx_data[idx_sample]] = 1
        else:
            idx_keep[idx_data] = 1

    scores = scores[idx_keep]
    spike_index = spike_index[idx_keep]

    return scores, spike_index
