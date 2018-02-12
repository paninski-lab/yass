import numpy as np
import logging

from yass.mfm import spikesort, suffStatistics, merge_move, cluster_triage


def run_cluster(scores, masks, groups, spike_times,
              channel_groups, channel_index,
              n_features, CONFIG):
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

    n_channel_groups = len(channel_groups)
    n_channels, n_neigh = channel_index.shape

    max_cluster_id = -1
    spike_train = np.zeros((0, 2), 'int32')
    for g in range(n_channel_groups):
        
        # channels in the group
        core_channels = channel_groups[g]
        # include all channels neighboring to core channels
        neigh_cores = np.unique(channel_index[core_cohannels])
        neigh_cores = neigh_channels[neigh_cores < n_channels]
        n_neigh_channels = neigh_cores.shape[0]
        
        # initialize data for this channel group
        score = np.zeros((0, n_features, n_neigh_channels))
        mask = np.zeros((0, n_neigh_channels))
        group = np.zeros(0, 'int32')
        spike_time = np.zeros((0, 2), 'int32')

        # gather information
        max_group_id = -1 
        for _, channel in enumerate(core_channels):
            if scores[channel].shape[0] > 0:
                
                # number of data
                n_data_channel = scores[channel].shape[0]
                # neighboring channels in this group
                neigh_channels = channel_index[channel][
                    channel_index[channel] < n_channels]

                #
                for j in range(neigh_channels.shape[0]):
                    score_temp[:, :, neigh_cores == neigh_channels[j]
                              ] = scores[channel][:, :, j]
                score = np.concatenate((score, score_temp), axis=0)
                mask = np.concatenate((mask, masks[channel]), axis=0)
                spike_time = np.concatenate((spike_time, spike_times[channel]),
                                            axis=0)
                group = npconcatenate((group, groups[channel] + max_group_id + 1),
                                      axis=0)
                max_group_id += np.max(groups[channel]) + 1

        if score.shape[0] > 0:

            # run clustering
            cluster_id = spikesort(score, mask, group, config)
            
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



def run_cluster_loccation(scores, spike_times, CONFIG):
    """
    Parameters
    ----------

    Returns
    -------
    spike_train:
        ?
    """
    
    logger = logging.getLogger(__name__)
    
    n_channels = len(scores)
    
    global_vbParam = None
    global_maskedData = None
    global_score = None
    global_spike_time = None
    for channel in range(n_channels):
        
        logger.info('Processing channel {}'.format(channel))
       
        score = scores[channel]
        spike_time = spike_times[channel]
        n_data = score.shape[0]
        
        if n_data > 0:
            
            # make a fake mask of ones to run clustering algorithm
            mask = np.ones((n_data, 1))
            group = np.arange(n_data)
            (vbParam, maskedData) = spikesort(score, mask,
                                              group, CONFIG)
            
            # gather clustering information into global variable
            (global_vbParam, global_maskedData,
             global_score, 
             global_spike_time) = global_cluster_info(vbParam,
                                                      maskedData,
                                                      score,
                                                      spike_time,
                                                      global_vbParam,
                                                      global_maskedData,
                                                      global_score,
                                                      global_spike_time)

    logger.info('merging all channels')

    # data info
    n_clusters = global_vbParam.muhat.shape[1]
    n_data = global_score.shape[0]

    # update local param with all data
    global_vbParam.update_local(global_maskedData)
    # update sufficient statistics with all data
    suffStat = suffStatistics(global_maskedData, global_vbParam)
    # run final merge
    global_vbParam, suffStat, _ = merge_move(global_maskedData,
                                             global_vbParam,
                                             suffStat,
                                             CONFIG,
                                             np.ones(n_clusters),
                                             1)

    # get final assignment
    # this is assignment per gorup
    assignment = np.argmax(global_vbParam.rhat, axis=1)

    # model based triage
    idx_triage = cluster_triage(global_vbParam, global_score, 3)
    assignment = assignment[~idx_triage]
    global_spike_time = global_spike_time[~idx_triage]

    # make spike train
    spike_train = np.hstack(
        (global_spike_time[:, np.newaxis], assignment[:, np.newaxis]))

    # sort based on spike_time
    idx_sort = np.argsort(spike_train[:, 0])

    return spike_train[idx_sort]


def global_cluster_info(vbParam, maskedData, score, spike_time,
                        global_vbParam=None, global_maskedData=None,
                        global_score=None, global_spike_time=None
                       ):
    
    if global_vbParam is None:
        global_vbParam = vbParam
        global_maskedData = maskedData
        global_score = score
        global_spike_time = spike_time
        
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
        
        # append maskedData
        global_maskedData.sumY = np.concatenate(
            [global_maskedData.sumY, maskedData.sumY],
            axis=0)
        global_maskedData.sumYSq = np.concatenate(
            [global_maskedData.sumYSq, maskedData.sumYSq],
            axis=0)
        global_maskedData.sumEta = np.concatenate(
            [global_maskedData.sumEta, maskedData.sumEta],
            axis=0)
        global_maskedData.weight = np.concatenate(
            [global_maskedData.weight, maskedData.weight],
            axis=0)
        global_maskedData.groupMask = np.concatenate(
            [global_maskedData.groupMask, maskedData.groupMask],
            axis=0)
        global_maskedData.meanY = np.concatenate(
            [global_maskedData.meanY, maskedData.meanY],
            axis=0)
        global_maskedData.meanYSq = np.concatenate(
            [global_maskedData.meanYSq, maskedData.meanYSq],
            axis=0)
        global_maskedData.meanEta = np.concatenate(
            [global_maskedData.meanEta, maskedData.meanEta],
            axis=0)
        
        # append score
        global_score = np.concatenate([global_score, score], axis=0)

        # append spike_time
        global_spike_time = np.concatenate([global_spike_time,
                                           spike_time], axis=0)

    return (global_vbParam, global_maskedData, global_score,
            global_spike_time)


    
