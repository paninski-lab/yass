import logging

import numpy as np
from ..geometry import order_channels_by_distance

logger = logging.getLogger(__name__)


def get_waveforms(recording, spike_index, proj, neighbors, geom, nnt, th):

    """Extract waveforms from detected spikes

    Parameters
    ----------
    recording: matrix (observations, number of channels)
        Multi-channel recordings
    spike_index: matrix (number of spikes, 2)
        Spike index matrix, as returned from any of the detectors
    proj: matrix (waveform temporal length, number of features)
        Projection matrix that reduces the dimension of waveform
    neighbors: matrix (number of channels, number of channel)
        Neighbors matrix
    geom: matrix (number of channels, 2)
        Each row is the x,y coordinate of each channel
    nnt: class
        Class for Neural Network based triage
    th: int
        Threshold for Neural Network triage algorithm

    Returns
    -------
    score: matrix (observations, number of features, number of neighbors)
    clear_spike: boolean vector (observations)
        Boolean indicating if it is a clear spike or not

    Notes
    -----
    Le'ts consider a single channel recording V, where V is a vector of
    length 1 x T. When a spike is detected at time t, then (V_(t-R),
    V_(t-R+1), ..., V_t, V_(t+1),...V_(t+R)) is going to be a waveform.
    (a small snippet from the recording around the spike time)
    """
    # column ids for index matrix
    SPIKE_TIME, MAIN_CHANNEL = 0, 1

    n_times, n_channels = recording.shape
    n_spikes, _ = spike_index.shape
    window_size, n_features = proj.shape
    spike_size = int((window_size-1)/2)
    nneigh = np.max(np.sum(neighbors, 0))

    recording = np.concatenate((recording, np.zeros((n_times,1))), axis=1)
    c_idx = np.ones((n_channels, nneigh), 'int32')*n_channels
    for c in range(n_channels):
        ch_idx, _ = order_channels_by_distance(c,
                                               np.where(neighbors[c])[0],
                                               geom)
        c_idx[c,:ch_idx.shape[0]] = ch_idx

    spike_index_clear = np.zeros((0,2), 'int32')
    spike_index_collision = np.zeros((0,2), 'int32')
    score = np.zeros((0, n_features, nneigh), 'float32')

    nbuff = 500000
    wf = np.zeros((nbuff, window_size, nneigh), 'float32')

    count = 0
    for j in range(n_spikes):
        t = spike_index[j,SPIKE_TIME]
        c = c_idx[spike_index[j,MAIN_CHANNEL]]
        wf[count] = recording[(t-spike_size):(t+spike_size+1),c]
        count += 1

        # when we gathered enough spikes, go through triage NN and save score
        if (count == nbuff) or (j == n_spikes -1):
            # if we seek all spikes before reaching the buffer size,
            # size of buffer becomes the number of leftover spikes
            if j == n_spikes-1:
                nbuff = count
                wf = wf[:nbuff]

            # going through triage NN.
            # The output is 1 for clear spike and 0 otherwise
            clear_spike = nnt.nn_triage(wf, th)

            # collect clear and colliding spikes
            spike_index_buff = spike_index[(j-nbuff+1):(j+1)]
            spike_index_clear = np.concatenate((spike_index_clear,
                spike_index_buff[clear_spike]))
            spike_index_collision = np.concatenate((spike_index_collision,
                spike_index_buff[~clear_spike]))

            # calculate score and collect into variable 'score'
            reshaped_wf = np.reshape(np.transpose(wf[clear_spike],(0,2,1)),
                (-1,window_size))
            score_temp = np.transpose(np.reshape(np.matmul(reshaped_wf, proj),
                (-1, nneigh, n_features)), (0,2,1))
            score = np.concatenate((score,score_temp), axis = 0)

            # set counter back to zero
            count = 0

    return spike_index_clear, score, spike_index_collision



def get_waveforms_depreciated(rec, neighbors, index, get_score, proj, spike_size,
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
