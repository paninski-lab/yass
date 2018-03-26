import numpy as np
import logging

from yass.mfm import spikesort
from yass import mfm


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
    global_score = None
    global_vbParam = None
    global_spike_time = None
    global_cluster_id = None

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
            cluster_id, vbParam = spikesort(score, mask,
                                            group, CONFIG)

            idx_triage = (cluster_id == -1)

            cluster_id = cluster_id[~idx_triage]
            spike_time = spike_time[~idx_triage]
            score = score[~idx_triage]

            # gather clustering information into global variable
            (global_vbParam,
             global_score, global_spike_time,
             global_cluster_id) = global_cluster_info(vbParam,
                                                      score,
                                                      spike_time,
                                                      cluster_id,
                                                      global_vbParam,
                                                      global_score,
                                                      global_spike_time,
                                                      global_cluster_id)

    # global merge
    maha = calculate_mahalanobis(global_vbParam)
    check = np.logical_or(maha < 15, maha.T < 15)
    while np.any(check):
        cluster = np.where(np.any(check, axis=1))[0][0]
        neigh_clust = list(np.where(check[cluster])[0])
        global_cluster_id, maha, merged = merge_move_patches(
            cluster, neigh_clust, global_score, global_cluster_id,
            global_vbParam, maha, CONFIG)
        check = np.logical_and(maha < 15, maha.T < 15)

    # clean empty spaces
    global_spike_time, global_cluster_id, global_score = clean_empty_cluster(
        global_spike_time, global_cluster_id, global_score[:, :, 0])

    # make spike train
    spike_train = np.hstack(
        (global_spike_time[:, np.newaxis],
         global_cluster_id[:, np.newaxis]))

    # sort based on spike_time
    idx_sort = np.argsort(spike_train[:, 0])

    return spike_train[idx_sort], global_score[idx_sort]


def calculate_mahalanobis(vbParam):
    diff = np.transpose(vbParam.muhat, [1, 2, 0]) - vbParam.muhat[..., 0].T
    clustered_prec = np.transpose(
        vbParam.Vhat[:, :, :, 0] * vbParam.nuhat, [2, 0, 1])
    maha = np.squeeze(np.matmul(diff[:, :, np.newaxis],
                                np.matmul(clustered_prec[:, np.newaxis],
                                          diff[..., np.newaxis])), axis=[2, 3])
    maha[np.diag_indices(maha.shape[0])] = np.inf

    return maha


def merge_move_patches(cluster, neigh_clusters, scores,
                       clusterid, vbParam, maha, cfg):

    while len(neigh_clusters) > 0:
        i = neigh_clusters[0]
        indices = np.logical_or(clusterid == cluster, clusterid == i)
        ka, kb = min(cluster, i), max(cluster, i)
        local_scores = scores[indices]
        local_vbParam = mfm.vbPar(None)
        local_vbParam.muhat = vbParam.muhat[:, [cluster, i]]
        local_vbParam.Vhat = vbParam.Vhat[:, :, [cluster, i]]
        local_vbParam.invVhat = vbParam.invVhat[:, :, [cluster, i]]
        local_vbParam.nuhat = vbParam.nuhat[[cluster, i]]
        local_vbParam.lambdahat = vbParam.lambdahat[[cluster, i]]
        local_vbParam.ahat = vbParam.ahat[[cluster, i]]
        mask = np.ones([local_scores.shape[0], 1])
        local_maskedData = mfm.maskData(local_scores, mask,
                                        np.arange(local_scores.shape[0]))
        local_vbParam.update_local(local_maskedData)
        local_suffStat = mfm.suffStatistics(local_maskedData, local_vbParam)

        ELBO = mfm.ELBO_Class(local_maskedData, local_suffStat,
                              local_vbParam, cfg)
        L = np.ones(2)
        (local_vbParam,
         local_suffStat,
         merged, _, _) = mfm.check_merge(local_maskedData,
                                         local_vbParam,
                                         local_suffStat, 0, 1,
                                         cfg, L, ELBO)

        if merged:
            vbParam.muhat = np.delete(vbParam.muhat, kb, 1)
            vbParam.muhat[:, ka] = local_vbParam.muhat[:, 0]

            vbParam.Vhat = np.delete(vbParam.Vhat, kb, 2)
            vbParam.Vhat[:, :, ka] = local_vbParam.Vhat[:, :, 0]

            vbParam.invVhat = np.delete(vbParam.invVhat, kb, 2)
            vbParam.invVhat[:, :, ka] = local_vbParam.invVhat[:, :, 0]

            vbParam.nuhat = np.delete(vbParam.nuhat, kb, 0)
            vbParam.nuhat[ka] = local_vbParam.nuhat[0]

            vbParam.lambdahat = np.delete(vbParam.lambdahat, kb, 0)
            vbParam.lambdahat[ka] = local_vbParam.lambdahat[0]

            vbParam.ahat = np.delete(vbParam.ahat, kb, 0)
            vbParam.ahat[ka] = local_vbParam.ahat[0]

            clusterid[indices] = ka
            clusterid[clusterid > kb] = clusterid[clusterid > kb] - 1
            neigh_clusters.pop()

            maha = np.delete(maha, kb, 1)
            maha = np.delete(maha, kb, 0)

            diff = vbParam.muhat[:, :, 0] - local_vbParam.muhat[:, :, 0]

            prec = local_vbParam.Vhat[..., 0] * local_vbParam.nuhat[0]
            maha[ka] = np.squeeze(
                np.matmul(
                    diff.T[:, np.newaxis, :],
                    np.matmul(prec[:, :, 0],
                              diff.T[..., np.newaxis])))

            prec = np.transpose(vbParam.Vhat[..., 0] *
                                vbParam.nuhat, [2, 0, 1])
            maha[:, ka] = np.squeeze(
                np.matmul(
                    diff.T[:, np.newaxis, :],
                    np.matmul(prec, diff.T[..., np.newaxis])))

            maha[ka, ka] = np.inf
            neigh_clusters = list(
                np.where(np.logical_and(maha[ka] < 5, maha.T[ka] < 5))[0])
            cluster = ka

        if not merged:
            maha[ka, kb] = maha[kb, ka] = np.inf
            neigh_clusters.pop()

    return clusterid, maha, merged


def global_cluster_info(vbParam, score, spike_time, cluster_id,
                        global_vbParam, global_score,
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
    if global_vbParam is None:
        global_vbParam = vbParam
        global_score = score
        global_spike_time = spike_time
        global_cluster_id = cluster_id
    else:

        # append global_vbParam
        global_vbParam.muhat = np.concatenate(
            [global_vbParam.muhat, vbParam.muhat], axis=1)
        global_vbParam.Vhat = np.concatenate(
            [global_vbParam.Vhat, vbParam.Vhat], axis=2)
        global_vbParam.invVhat = np.concatenate(
            [global_vbParam.invVhat, vbParam.invVhat],
            axis=2)
        global_vbParam.lambdahat = np.concatenate(
            [global_vbParam.lambdahat, vbParam.lambdahat],
            axis=0)
        global_vbParam.nuhat = np.concatenate(
            [global_vbParam.nuhat, vbParam.nuhat],
            axis=0)
        global_vbParam.ahat = np.concatenate(
            [global_vbParam.ahat, vbParam.ahat],
            axis=0)

        # append score
        global_score = np.concatenate([global_score, score], axis=0)

        # append spike_time
        global_spike_time = np.hstack((global_spike_time,
                                       spike_time))

        # append assignment
        cluster_id_max = np.max(global_cluster_id)
        global_cluster_id = np.hstack([
            global_cluster_id,
            cluster_id + cluster_id_max + 1])

    return (global_vbParam, global_score,
            global_spike_time, global_cluster_id)


def clean_empty_cluster(spike_time, cluster_id, score, max_spikes=20):

    n_units = np.max(cluster_id) + 1
    units_keep = np.zeros(n_units, 'bool')
    for k in range(n_units):
        if np.sum(cluster_id == k) >= max_spikes:
            units_keep[k] = 1

    Ks = np.where(units_keep)[0]
    spike_time_clean = np.zeros(0, 'int32')
    cluster_id_clean = np.zeros(0, 'int32')
    score_clean = np.zeros((0, score.shape[1]))
    for j, k in enumerate(Ks):

        spt_temp = spike_time[cluster_id == k]
        score_temp = score[cluster_id == k]

        spike_time_clean = np.hstack((spike_time_clean,
                                      spt_temp))
        cluster_id_clean = np.hstack((cluster_id_clean,
                                      np.ones(spt_temp.shape[0], 'int32')*j))
        score_clean = np.concatenate((score_clean,
                                      score_temp), 0)

    idx_sort = np.argsort(spike_time_clean)

    return (spike_time_clean[idx_sort],
            cluster_id_clean[idx_sort],
            score_clean[idx_sort])
