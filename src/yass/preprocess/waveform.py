import logging

import numpy as np
from ..geometry import order_channels_by_distance

logger = logging.getLogger(__name__)


def get_waveforms(rec, neighbors, index, get_score, proj, spike_size,
                  n_features, geom, nnt, th):

    """Extract waveforms from detected spikes

    Parameters
    ----------
    recordings: matrix (observations, number of channels)
        Multi-channel recordings
    neighbors: matrix (number of channels, number of channel)
        Neighbors matrix
    index: matrix (number of spikes, 3)
        Spike index matrix, as returned from any of the detectors
    get_score: bool
        Whether or not to calculate scores, if False returns an array with 0s
    proj:
        ?
    spike_size:
        ?
    n_features:
        ?
    n_features:
        ?

    Returns
    -------
    score: list
        List with n_channels elements, each element contains the data whose
        main channel is the i channel where i is the index in the list
    clr_index: list
        List with n_channels elements, each element contains the indexes
        for the spikes indicating whether it was a clear spike or not
    spike_times: list
        List with n_channels elements, each element contains the spike time

    Notes
    -----
    Le'ts consider a single channel recording V, where V is a vector of
    length 1 x T. When a spike is detected at time t, then (V_(t-R),
    V_(t-R+1), ..., V_t, V_(t+1),...V_(t+R)) is going to be a waveform.
    (a small snippet from the recording around the spike time)
    """
    # column ids for index matrix
    SPIKE_TIME, MAIN_CHANNEL = 0, 1

    R = spike_size
    _, n_channels = rec.shape
    score = list()
    clr_idx = list()
    spike_time = list()

    # loop over every channel
    for c in range(n_channels):

        # get spikes whose main channel is the current channel
        idx = index[:, MAIN_CHANNEL] == c

        # check if there is at least one spike for the current channel
        nc = np.sum(idx)

        # get the indices for channel c neighbors
        (ch_idx, ) = np.where(neighbors[c])

        # if there are spikes for channel c, process them...
        if nc > 0:

            # get spike times
            spike_time_c = index[idx, SPIKE_TIME]

            # append to spike_times
            spike_time.append(spike_time_c)

            if get_score == 1:
                # get waveforms
                wf = np.zeros((nc, 2*R+1, ch_idx.shape[0]))

                for j in range(nc):
                    wf[j] = rec[spike_time_c[j]+np.arange(-R, R+1)][:, ch_idx]

                temp, c_order = order_channels_by_distance(c, ch_idx, geom)
                clr_idx_c = nnt.nn_triage(wf[:,:,c_order], th)
                nc_clear = np.sum(clr_idx_c)

            else:
                nc_clear = 0

            if nc_clear > 0:
                clr_idx.append(np.where(clr_idx_c)[0])

                # get score
                wf = wf[clr_idx_c]
                score.append(np.swapaxes(np.matmul(np.reshape(np.swapaxes(wf, 1, 2), (-1, 2*R+1)), proj)
                                         .reshape((wf.shape[0], wf.shape[2], -1)), 1, 2))
            else:
                logger.debug('Spikes detected with main channel {c}, '
                             'but get_score is False, returning zeros in '
                             'score and clr_idx...'.format(c=c))
                clr_idx.append(np.zeros(0, 'int32'))
                score.append(np.zeros((0, n_features, np.sum(ch_idx))))

        # otherwise return zeros...
        else:
            logger.debug('No spikes detected with main channel {c}, '
                         'returning zeros...'.format(c=c))
            spike_time.append(np.zeros(0, 'int32'))
            clr_idx.append(np.zeros(0, 'int32'))
            score.append(np.zeros((0, n_features, np.sum(ch_idx))))

    return score, clr_idx, spike_time
