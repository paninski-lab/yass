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
    n_channels = len(scores)

    for c in range(n_channels):
        n_data = scores[c].shape[0]
        if n_data > n_sample:

            idx_keep = np.random.choice(n_data,
                                        n_sample,
                                        replace=False)
            scores[c] = scores[c][idx_keep]
            spike_index[c] = spike_index[c][idx_keep]

    return scores, spike_index
