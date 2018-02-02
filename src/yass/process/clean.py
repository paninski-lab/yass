import numpy as np


# TODO: missing documentation
# FIXME: doesnt seem like this is used at all
def clean_output(spike_train_clear, spt, clr_idx, batch_size, buff):
    """[Description]

    Parameters
    ----------

    Returns
    -------
    """
    spikeTrain = np.zeros((spike_train_clear.shape[0], 2), 'int32')
    spikeTrain[:, 1] = spike_train_clear[:, 0]

    spt_left = list()
    for c in range(len(spt)):

        # from spike_train_clear, get index of of channel c
        idx_c = spike_train_clear[:, 1] == c

        # get spike time from spt
        clr_idx_c = spike_train_clear[idx_c, 2]
        spt_clear = spt[c][clr_idx_c]

        # get actual spike time using batch number, spike time in each batch,
        # and buffer size
        spikeTrain[idx_c, 0] = spt_clear[:, 0] + \
            batch_size*spt_clear[:, 1] - buff

        # save unsorted spike times in spt_left
        idx_col = np.ones(spt[c].shape[0], 'bool')
        idx_col[clr_idx_c] = 0
        spt_left.append(spt[c][idx_col])

    return spikeTrain, spt_left
