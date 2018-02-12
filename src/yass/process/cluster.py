import numpy as np

from yass.mfm import spikesort, suffStatistics, merge_move, cluster_triage


def runSorter(scores, masks, groups, spike_times,
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
                (spike_time, cluster_id + max_cluster_id + 1)))
            
            max_cluster_id += np.max(cluster_id) + 1

    # sort based on spike_time
    idx_sort = np.argsort(spike_train[:, 0])

    return spike_train[idx_sort]



def runSorter_loccation(scores, groups, spike_times,
                        n_features, CONFIG):
    """
    Parameters
    ----------

    Returns
    -------
    spike_train:
        ?
    """
    
    n_channels = len(scores)
    
    vbParam_global = None
    maskedData_global = None
    score_global = None
    group_global = None
    spike_time_global = None
    for channel in range(n_channels):
        
        logger.info('Processing channel {}'.format(i))
       
        score = scores[channel]
        group = groups[channel]
        spike_time = spike_times[channel]
        n_data = scores.shape[0]
        
        if n_data > 0:
            
            # make a fake mask of ones to run clustering algorithm
            mask = np.ones((n_data, 1))
            (vbParam, maskedData) = spikesort(score, mask,
                                              group, CONFIG)
            
            # gather clustering information into global variable
            (vbParam_global, maskedData_global,
             score_global, group_global, 
             spike_time_global) = global_cluster_info(vbParam,
                                                      maskedData,
                                                      score,
                                                      group,
                                                      spike_time,
                                                      vbParam_global,
                                                      maskedData_global,
                                                      score_global,
                                                      group_global,
                                                      spike_time_global
                                                     )

    logger.info('merging all channels')

    # data info
    n_clusters = global_vbParam.muhat.shape[1]
    n_data = score_global.shape[0]

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
    assignment_per_group = np.argmax(global_vbParam.rhat, axis=1)
    # make it into assignmnet per data
    assignment = np.zeros(n_data, 'int16')
    for j in range(n_data):
        assignment[j] = assignment_per_data[
            group_global[j]]

        # model based triage
        idx_triage = cluster_triage(global_vbParam, score_global, 3)
        assignment = assignment[~idx_triage]
        spike_time_global = spike_time_global[~idx_triage]

        # make spike train
        spike_train_clear = np.hstack(
            (spike_time_global, assignment), axis=1)

        # sort based on spike_time
        idx_sort = np.argsort(spike_train[:, 0])

        return spike_train[idx_sort]



def global_cluster_info(vbParam, maskedData, score, group, spike_time,
                        vbParam_global=None, maskedData_global=None,
                        score_global=None, group_global=None,
                        spike_time_global=None
                       ):
    
    if vbParam_global is None:
        vbParam_global = vbParam
        maskedData_global = maskedData
        score_global = score
        group_global = group
        
    else:
        
        # append global_vbParam
        global_vbParam.muhat = np.concatenate(
            [global_vbParam.muhat, vbParam.muhat], axis=1)
        global_vbParam.Vhat = np.concatenate(
            [global_vbParam.Vhat, vbParam.Vhat], axis=2)
        global_vbParam.invVhat = np.concatenate(
            [global_vbParam.invVhat, local_vbParam.invVhat],
            axis=2)
        global_vbParam.lambdahat = np.concatenate(
            [global_vbParam.lambdahat, local_vbParam.lambdahat],
            axis=0)
        global_vbParam.nuhat = np.concatenate(
            [global_vbParam.nuhat, local_vbParam.nuhat],
            axis=0)
        global_vbParam.ahat = np.concatenate(
            [global_vbParam.ahat, local_vbParam.ahat],
            axis=0)
        
        # append maskedData
        global_maskedData.sumY = np.concatenate(
            [global_maskedData.sumY, local_maskedData.sumY],
            axis=0)
        global_maskedData.sumYSq = np.concatenate(
            [global_maskedData.sumYSq, local_maskedData.sumYSq],
            axis=0)
        global_maskedData.sumEta = np.concatenate(
            [global_maskedData.sumEta, local_maskedData.sumEta],
            axis=0)
        global_maskedData.weight = np.concatenate(
            [global_maskedData.weight, local_maskedData.weight],
            axis=0)
        global_maskedData.groupMask = np.concatenate(
            [global_maskedData.groupMask, local_maskedData.groupMask],
            axis=0)
        global_maskedData.meanY = np.concatenate(
            [global_maskedData.meanY, local_maskedData.meanY],
            axis=0)
        global_maskedData.meanYSq = np.concatenate(
            [global_maskedData.meanYSq, local_maskedData.meanYSq],
            axis=0)
        global_maskedData.meanEta = np.concatenate(
            [global_maskedData.meanEta, local_maskedData.meanEta],
            axis=0)
        
        # append score
        score_global = np.concatenate([score_global, score], axis=0)

        # append group
        global_max_id = np.max(group_global)
        group_global = np.concatenate([group_global, group+global_max_id+1])
        
        # append spike_time
        spike_time_global = np.concatenate([spike_time_global,
                                           spike_time], axis=0)

        return (global_vbParam, global_maskedData, score_global,
                group_global, spike_time_global)


    
