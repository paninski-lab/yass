"""
Blocks for noise processing
"""
import numpy as np

from ... import read_config
from ...preprocess.filter import butterworth
from ...geometry import (n_steps_neigh_channels,
                         order_channels_by_distance)
from ...preprocess import standarize
from . import util


# FIXME: i dont think is is being used
def covariance(recordings, temporal_size, neigbor_steps):
    """Compute noise spatial and temporal covariance

    Parameters
    ----------
    recordings: matrix
        Multi-cannel recordings (n observations x n channels)
    temporal_size:

    neigbor_steps: int
        Number of steps from the multi-channel geometry to consider two
        channels as neighors
    """
    CONFIG = read_config()

    # get the neighbor channels at a max "neigbor_steps" steps
    neigh_channels = n_steps_neigh_channels(CONFIG.neighChannels,
                                            neigbor_steps)

    # sum neighor flags for every channel, this gives the number of neighbors
    # per channel, then find the channel with the most neighbors
    # TODO: why are we selecting this one?
    channel = np.argmax(np.sum(neigh_channels, 0))

    # get the neighbor channels for "channel"
    (neighbords_idx,) = np.where(neigh_channels[channel])

    # order neighbors by distance
    neighbords_idx, temp = order_channels_by_distance(channel, neighbords_idx,
                                                      CONFIG.geom)

    # from the multi-channel recordings, get the neighbor channels
    # (this includes the channel with the most neighbors itself)
    rec = recordings[:, neighbords_idx]

    # filter recording
    if CONFIG.preprocess.filter == 1:
        rec = butterworth(rec, CONFIG.filter.low_pass_freq,
                          CONFIG.filter.high_factor,
                          CONFIG.filter.order,
                          CONFIG.recordings.sampling_rate)

    # standardize recording
    sd_ = standarize.sd(rec, CONFIG.recordings.sampling_rate)
    rec = standarize.standarize(rec, sd_)

    # compute and return spatial and temporal covariance
    return util.covariance(rec, temporal_size, neigbor_steps,
                           CONFIG.spikeSize)
