def make_list(scores, spike_index, n_channels):
    """
    Change a data structure of spike_index and score from an array of two
    columns to a list
    Do the same thing to scores too

    Parameters
    ----------
    scores: np.ndarray(n_spikes, n_features, n_neigh)
       A 3D array storing scores of each spike

    spike_index: numpy.ndarray (n_spikes, 2)
       A 2D array containing spikes information with two columns,
       where the first column is spike time and the second is channel.

    n_channels: int
       the number of channels in recording (or length of output list)

    Returns
    -------
    scores_list: list (n_channels)
        A list such that scores_list[c] contains all scores whose main
        channel is c

    spike_index_list: list (n_channels)
        A list such that spike_index_list[c] cointains all spike times
        whose channel is c
    """

    # initialize list
    scores_list = [None]*n_channels
    spike_index_list = [None]*n_channels

    for c in range(n_channels):
        idx = spike_index[:, 1] == c
        scores_list[c] = scores[idx]
        spike_index_list[c] = spike_index[idx, 0]

    return scores_list, spike_index_list
