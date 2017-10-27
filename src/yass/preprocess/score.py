"""Score related functions
"""
import os
import numpy as np


def getScore(spt, rot, n_channels, spike_size, n_features, neighbors,
             wf_path, scale_to_save, n_batches, n_portions, buff,
             batch_size):
    """PCA scoring
    """
    wf_file = open(os.path.join(wf_path), 'rb')

    batch_size = batch_size+2*buff

    flattenedLength = 2*batch_size*n_channels

    neighchan = neighbors
    C = n_channels
    R = spike_size

    score = [None]*C
    clr_idx = [None]*C

    nBatches = n_batches
    nPortion = n_portions

    for i in range(0, nBatches):
        if i <= nPortion:

            wf_file.seek(flattenedLength*i)
            wrec = wf_file.read(flattenedLength)

            wrec = np.fromstring(wrec, dtype='int16')

            wrec = np.reshape(wrec, (-1, n_channels))
            wrec = wrec.astype('float32')/scale_to_save

            for c in range(C):
                ch_idx = np.where(neighchan[c])[0]
                clr_idx_temp = np.where(spt[c][:, 1] == i)[0]
                spt_c = spt[c][clr_idx_temp, 0]

                if spt_c.shape[0] > 0:
                    # get waveforms
                    wf = np.zeros((spt_c.shape[0], 2*R+1, ch_idx.shape[0]))

                    for j in range(spt_c.shape[0]):
                        wf[j] = wrec[spt_c[j]+np.arange(-R, R+1)][:, ch_idx]

                    score_temp = np.zeros(
                        (wf.shape[0], n_features, wf.shape[2]))
                    for j in range(ch_idx.shape[0]):
                        score_temp[:, :, j] = np.matmul(
                            wf[:, :, j], rot[:, :, ch_idx[j]])

                    if i == 0:
                        score[c] = score_temp
                        clr_idx[c] = clr_idx_temp
                    else:
                        score[c] = np.concatenate((score[c], score_temp))
                        clr_idx[c] = np.concatenate((clr_idx[c], clr_idx_temp))
                else:
                    if i == 0:
                        score[c] = np.zeros((0, n_features, ch_idx.shape[0]))
                        clr_idx[c] = np.zeros((0), 'int16')

    wf_file.close()

    return score, clr_idx


def getPcaSS(recordings, spike_times, spike_size, buff):
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
    n_obs, n_channels = recordings.shape
    window_idx = range(-spike_size, spike_size+1)
    window_size = len(window_idx)

    ss = np.zeros((window_size, window_size, n_channels))
    spikes_per_channel = np.zeros(n_channels)

    # iterate over every channel
    for c in range(n_channels):
        # get spikes times for the current channel
        channel_spike_times = spike_times[c]
        channel_spike_times = channel_spike_times[np.logical_and(
                              (channel_spike_times > spike_size+buff),
                              (channel_spike_times < n_obs-spike_size-buff))]

        channel_spikes = len(channel_spike_times)

        # create zeros matrix (window size x number of spikes for this channel)
        wf_temp = np.zeros((window_size, channel_spikes))

        # iterate over the window size
        for j in range(window_size):
            # fill in recording values for each spike time
            wf_temp[j, :] = recordings[channel_spike_times + window_idx[j], c]

        ss[:, :, c] = np.matmul(wf_temp, wf_temp.transpose())

        spikes_per_channel[c] = channel_spikes

    spikes_per_channel = spikes_per_channel.astype('int')

    return ss, spikes_per_channel


def getPCAProjection(ss, spikes_per_channel, n_features, neighbors):
    """Get PCA projection matrix per channel

    Parameters
    ----------
    ss: matrix
        SS matrix as returned from getPcaSS
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
