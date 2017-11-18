"""Score related functions
"""
import os
import numpy as np
from ..geometry import order_channels_by_distance


def get_score_pca(spike_index, rot, neighbors, geom, batch_size, BUFF, nBatches,
             wf_path, scale_to_save):
    """PCA scoring
    """
    # column ids for index matrix
    SPIKE_TIME, MAIN_CHANNEL = 0, 1

    window_size, n_features, n_channels = rot.shape
    spike_size = int((window_size-1)/2)
    n_spikes = spike_index.shape[0]

    wf_file = open(os.path.join(wf_path), 'rb')
    flattenedLength = 2*(batch_size + 2*BUFF)*n_channels

    nneigh = np.max(np.sum(neighbors, 0))
    c_idx = np.ones((n_channels, nneigh), 'int32')*n_channels
    for c in range(n_channels):
        ch_idx, _ = order_channels_by_distance(c,
                                               np.where(neighbors[c])[0],
                                               geom)
        c_idx[c,:ch_idx.shape[0]] = ch_idx

    score = np.zeros((n_spikes, n_features, nneigh), 'float32')

    counter_batch = 0
    for i in range(nBatches):
        idx_batch = np.logical_and(spike_index[:,0] > batch_size*i, 
                                   spike_index[:,0] < batch_size*(i+1))
        
        spike_index_batch = spike_index[idx_batch]
        spike_index_batch[:,0] = spike_index_batch[:,0] - batch_size*i + BUFF
        n_spikes_batch = spike_index_batch.shape[0]

        wf_file.seek(flattenedLength*i)
        wrec = wf_file.read(flattenedLength)
        wrec = np.fromstring(wrec, dtype='int16')
        wrec = np.reshape(wrec, (-1, n_channels))
        wrec = wrec.astype('float32')/scale_to_save
        wrec = np.concatenate((wrec, np.zeros((batch_size + 2*BUFF,1))), axis=1)

        nbuff = 50000
        wf = np.zeros((nbuff, window_size, nneigh), 'float32')
        count = 0
        for j in range(n_spikes_batch):
            t = spike_index_batch[j,SPIKE_TIME]
            ch_idx = c_idx[spike_index_batch[j,MAIN_CHANNEL]]
            wf[count] = wrec[(t-spike_size):(t+spike_size+1),ch_idx]
            count += 1

            if (count == nbuff) or (j == n_spikes_batch -1):
                # if we seek all spikes before reaching the buffer size,
                # size of buffer becomes the number of leftover spikes
                if j == n_spikes_batch-1:
                    nbuff = count
                    wf = wf[:nbuff]

                # calculate score and collect into variable 'score'
                score_temp = np.zeros((wf.shape[0],n_features, nneigh))
                for j in range(nneigh):
                    if ch_idx[j] < n_channels:
                        score_temp[:,:,j] = np.matmul(wf[:,:,j],rot[:,:,ch_idx[j]])
                score[counter_batch:(counter_batch+nbuff)] = score_temp

                # set counter back to zero
                count = 0
                counter_batch += nbuff
    wf_file.close()

    return score


def get_pca_suff_stat(recordings, spike_index, spike_size):
    """Get PCA SS matrix per recording channel

    Parameters
    ----------
    recordings: matrix [observations, number of channels]
        Multi-channel recordings
    spike_times:
        List with spike times, one element per channel
    spike_size: int
        Spike size
    buff:
        Buffer size
    """
    # column ids for index matrix
    SPIKE_TIME, MAIN_CHANNEL = 0, 1

    n_obs, n_channels = recordings.shape
    window_idx = range(-spike_size, spike_size+1)
    window_size = len(window_idx)

    pca_suff_stat = np.zeros((window_size, window_size, n_channels))
    spikes_per_channel = np.zeros(n_channels, 'int32')

    # iterate over every channel
    for c in range(n_channels):
        # get spikes times for the current channel
        channel_spike_times = spike_index[
            spike_index[:,MAIN_CHANNEL]==c, SPIKE_TIME]
        channel_spike_times = channel_spike_times[np.logical_and(
                              (channel_spike_times > spike_size),
                              (channel_spike_times < n_obs-spike_size-1))]

        channel_spikes = len(channel_spike_times)

        # create zeros matrix (window size x number of spikes for this channel)
        wf_temp = np.zeros((window_size, channel_spikes))

        # iterate over the window size
        for j in range(window_size):
            # fill in recording values for each spike time
            wf_temp[j, :] = recordings[channel_spike_times + window_idx[j], c]

        pca_suff_stat[:, :, c] = np.matmul(wf_temp, wf_temp.T)

        spikes_per_channel[c] = channel_spikes

    return pca_suff_stat, spikes_per_channel


def get_pca_projection(ss, spikes_per_channel, n_features, neighbors):
    """Get PCA projection matrix per channel

    Parameters
    ----------
    ss: matrix
        SS matrix as returned from get_pca_suff_stat
    spikes_per_channel: array
        Number of spikes per channel
    n_features: int
        Number of features
    neighbors: matrix
        Neighbors matrix
    """
    window_size, _, n_channels = ss.shape
    # allocate rotation matrix for each channel
    rot = np.zeros((window_size, n_features, n_channels))

    ss_all = np.sum(ss, 2)
    w, v = np.linalg.eig(ss_all)
    rot_all = v[:, np.argsort(w)[window_size:(window_size-n_features-1):-1]]

    for c in range(n_channels):
        if spikes_per_channel[c] <= window_size:
            if np.sum(spikes_per_channel[neighbors[c, :]]) <= window_size:
                rot[:, :, c] = rot_all
            else:
                w, v = np.linalg.eig(np.sum(ss[:, :, neighbors[c, :]], 2))
                rot[:, :, c] = v[:, np.argsort(w)[window_size:(window_size-n_features-1):-1]]
        else:
            w, v = np.linalg.eig(ss[:, :, c])
            rot[:, :, c] = v[:, np.argsort(w)[window_size:(window_size-n_features-1):-1]]

    return rot