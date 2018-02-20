import numpy as np
import logging

from yass.mfm import spikesort


def run_cluster(scores, masks, groups, spike_times,
                channel_groups, channel_index,
                n_features, CONFIG):
    """
    run clustering algorithm using MFM

    Parameters
    ----------
    scores: list (n_channels)
        A list such that scores[c] contains all scores whose main
        channel is c

    masks: list (n_channels)
        mask for each data in scores
        masks[c] is the mask of spikes in scores[c]

    groups: list (n_channels)
        coreset represented as group id.
        groups[c] is the group id of spikes in scores[c]

    spike_index: list (n_channels)
        A list such that spike_index[c] cointains all spike times
        whose channel is c

    channel_groups: list (n_channel_groups)
        Using divide-and-conquer approach, data will be split
        based on main channel. As an example, data in group g
        will be data whose main channel is one of channel_groups[g]

    channel_index: np.array (n_channels, n_neigh)
        neighboring channel information
        channel_index[c] contains the index of neighboring channels of
        channel c

    n_features: int
       number of features in each data per channel

    CONFIG: class
       configuration class

    Returns
    -------
    spike_train: np.array (n_data, 2)
        spike_train such that spike_train[j, 0] and spike_train[j, 1]
        are the spike time and spike id of spike j
    """

    # FIXME: mutating parameter
    # this function is passing a config object and mutating it,
    # this is not a good idea as having a mutable object lying around the code
    # can break things and make it hard to debug
    # (09/27/17) Eduardo

    n_channel_groups = len(channel_groups)
    n_channels, n_neigh = channel_index.shape

    # biggest cluster id is -1 since there is no cluster yet
    max_cluster_id = -1
    spike_train = np.zeros((0, 2), 'int32')
    for g in range(n_channel_groups):

        # channels in the group
        core_channels = channel_groups[g]
        # include all channels neighboring to core channels
        neigh_cores = np.unique(channel_index[core_channels])
        neigh_cores = neigh_cores[neigh_cores < n_channels]
        n_neigh_channels = neigh_cores.shape[0]

        # initialize data for this channel group
        score = np.zeros((0, n_features, n_neigh_channels))
        mask = np.zeros((0, n_neigh_channels))
        group = np.zeros(0, 'int32')
        spike_time = np.zeros((0), 'int32')

        # gather information
        max_group_id = -1
        for _, channel in enumerate(core_channels):
            if scores[channel].shape[0] > 0:

                # number of data
                n_data_channel = scores[channel].shape[0]
                # neighboring channels in this group
                neigh_channels = channel_index[channel][
                    channel_index[channel] < n_channels]

                # expand the number of channels and
                # re-organize data to match it
                score_temp = np.zeros((n_data_channel, n_features,
                                       n_neigh_channels))
                mask_temp = np.zeros((n_data_channel,
                                      n_neigh_channels))
                for j in range(neigh_channels.shape[0]):
                    c_idx = neigh_cores == neigh_channels[j]
                    score_temp[:, :, c_idx
                               ] = scores[channel][:, :, [j]]

                    mask_temp[:, c_idx] = masks[channel][:, [j]]

                # collect all data in this group
                score = np.concatenate((score, score_temp), axis=0)
                mask = np.concatenate((mask, mask_temp), axis=0)
                spike_time = np.concatenate((spike_time, spike_times[channel]),
                                            axis=0)
                group = np.concatenate((group,
                                        groups[channel] + max_group_id + 1),
                                       axis=0)
                max_group_id += np.max(groups[channel]) + 1

        if score.shape[0] > 0:

            # run clustering
            cluster_id = spikesort(score, mask, group, CONFIG)

            # model based triage
            idx_triage = cluster_id == -1
            cluster_id = cluster_id[~idx_triage]
            spike_time = spike_time[~idx_triage]

            spike_train = np.vstack((spike_train, np.hstack(
                (spike_time[:, np.newaxis],
                 cluster_id[:, np.newaxis] + max_cluster_id + 1))))

            max_cluster_id += (np.max(cluster_id) + 1)

    # sort based on spike_time
    idx_sort = np.argsort(spike_train[:, 0])

    return spike_train[idx_sort]


def run_cluster_location(scores, spike_times, CONFIG):
    """
    run clustering algorithm using MFM and location features

    Parameters
    ----------
    scores: list (n_channels)
        A list such that scores[c] contains all scores whose main
        channel is c

    spike_times: list (n_channels)
        A list such that spike_index[c] cointains all spike times
        whose channel is c

    CONFIG: class
        configuration class

    Returns
    -------
    spike_train: np.array (n_data, 2)
        spike_train such that spike_train[j, 0] and spike_train[j, 1]
        are the spike time and spike id of spike j
    """
    logger = logging.getLogger(__name__)

    n_channels = len(scores)
    global_spike_time = np.zeros(0).astype('uint16')
    global_cluster_id = np.zeros(0).astype('uint16')

    # run clustering algorithm per main channel
    for channel in range(n_channels):

        logger.info('Processing channel {}'.format(channel))

        score = scores[channel]
        spike_time = spike_times[channel]
        n_data = score.shape[0]

        if n_data > 0:

            # make a fake mask of ones to run clustering algorithm
            mask = np.ones((n_data, 1))
            group = np.arange(n_data)
            cluster_id = spikesort(score, mask,
                                   group, CONFIG)

            idx_triage = (cluster_id == -1)

            cluster_id = cluster_id[~idx_triage]
            spike_time = spike_time[~idx_triage]

            # gather clustering information into global variable
            (global_spike_time,
             global_cluster_id) = global_cluster_info(spike_time,
                                                      cluster_id,
                                                      global_spike_time,
                                                      global_cluster_id)

    # make spike train
    spike_train = np.hstack(
        (global_spike_time[:, np.newaxis],
         global_cluster_id[:, np.newaxis]))

    # sort based on spike_time
    idx_sort = np.argsort(spike_train[:, 0])

    return spike_train[idx_sort]


def global_cluster_info(spike_time, cluster_id,
                        global_spike_time, global_cluster_id):
    """
    Gather clustering information from each run

    Parameters
    ----------
    vbParam, maskedData: class
        cluster information output from MFM

    score: np.array (n_data, n_features, 1)
        score used for each clustering

    spike_time: np.array (n_data, 1)
        spike time that matches with each score

    global_vbParam, global_maskedData: class
        a class that contains cluster information from all
        previous run,

    global_score: np.array (n_data_all, n_features, 1)
        all scores from previous runs

    global_spike_times: np.array (n_data_all, 1)
        spike times matched to global_score

    global_cluster_id: np.array (n_data_all, 1)
        cluster id matched to global_score

    Returns
    -------
    global_vbParam, global_maskedData: class
        a class that contains cluster information after
        adding the current one

    global_score: np.array (n_data_all, n_features, 1)
        all scores after adding the current one

    global_spike_times: np.array (n_data_all, 1)
        spike times matched to global_score

    global_cluster_id: np.array (n_data_all, 1)
        cluster id matched to global_score
    """
    # append spike_time
    global_spike_time = np.concatenate([global_spike_time,
                                       spike_time], axis=0)

    # append assignment
    if global_cluster_id.size == 0:
        cluster_id_max = -1
    else:
        cluster_id_max = np.max(global_cluster_id)
    global_cluster_id = np.hstack([
        global_cluster_id,
        cluster_id + cluster_id_max + 1])

    return (global_spike_time, global_cluster_id)
