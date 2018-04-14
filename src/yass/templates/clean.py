import numpy as np


def clean_up_templates(templates, weights, spike_train,
                       tmp_loc, geometry, neighbors,
                       snr_threshold, spread_threshold):
    """Clean up bad templates

    Parameters
    ----------
    templates: numpy.ndarray(n_channels, temporal_size, n_templates)
        templates

    weights: np.array(n_templates)
        weights coming out of template computation

    spike_train: np.array(n_data, 3)
        The 3 columns represent spike time, unit id,
        weight (from soft assignment)

    tmp_loc: np.array(n_templates)
        At which channel the clustering is done.

    geometry: np.array(n_channels, 2)
        geometry info

    neighbors: np.array(n_channels, n_channels) boolean
        neighboring channel info

    snr_threshold: float
        a threshold for removing small template

    spread_threshold: float
        a threshold for removing widely spread templates


    Returns
    -------
    templates: npy.ndarray
        Templates after clean up

    weights: np.array(n_templates)
        weights after clean up

    spike_train2: np.array
        spike_train after clean up

    idx_good_templates: np.array
        index of which templates are kept
    """
    # get size
    n_channels, temporal_size, n_templates = templates.shape

    # get energy
    energy = np.ptp(templates, axis=1)
    mainc = np.argmax(energy, axis=0)

    # check for overly spread template first
    too_spread = np.zeros(n_templates, 'bool')
    uncentered = np.zeros(n_templates, 'bool')
    too_small = np.zeros(n_templates, 'bool')
    for k in range(n_templates):

        # checking for spread
        idx_c = energy[:, k] > np.max(energy[:, k]) * 0.5
        if np.sum(idx_c) > 1:
            lam, V = np.linalg.eig(
                np.cov(geometry[idx_c].T, aweights=energy[idx_c, k]))
            lam[lam < 0] = 0
            if np.sqrt(np.max(lam)) > spread_threshold:
                too_spread[k] = 1

        # checking for uncentered
        ch_idx = np.where(neighbors[mainc[k]])[0]
        if not np.any(tmp_loc[k] == ch_idx):
            uncentered[k] = 1

        # checking for small templates
        if energy[mainc[k], k] < snr_threshold:
            too_small[k] = 1

    idx_good_templates = np.where(~np.logical_or(
        np.logical_or(too_spread, uncentered), too_small))[0]

    spike_train2 = np.zeros((0, 3), 'int32')
    for j, k in enumerate(idx_good_templates):
        idx_k = np.where(spike_train[:, 1] == k)[0]
        temp = np.copy(spike_train[idx_k])
        temp[:, 1] = j
        spike_train2 = np.vstack((spike_train2, temp))

    templates = templates[:, :, idx_good_templates]
    weights = weights[idx_good_templates]

    return templates, weights, spike_train2, idx_good_templates
