import progressbar
import numpy as np
import datetime as dt

from yass.MFM import spikesort


def runSorter(scores, masks, groups, spike_times,
              channel_index, CONFIG)

    """
    Parameters
    ----------

    Returns
    -------
    spike_train:
        ?
    """
    # FIXME: mutating parameter
    # this function is passing a config object and mutating it,
    # this is not a good idea as having a mutable object lying around the code
    # can break things and make it hard to debug
    # (09/27/17) Eduardo

    n_channels = len(scores)
    
    cluster_ids = [None]*n_channels
    for channel in range(n_channels):
        clsuter_id = spikesort(scores[channel], masks[channel], groups[channel],
                               config)

        idx_keep = cluster_id > 0
        cluster_ids[channel] = cluster_id[idx_keep]
        spike_times[channel] = spike_times[channel][idex_keep]
        scores[channel] scores[channel][idx_keep]

    return spike_train