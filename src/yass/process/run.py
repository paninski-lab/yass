"""
Process pipeline: triage (optional) coreset (optional), masking, clustering,
getting templates and cleaning
"""
import numpy as np
import os
import logging
import datetime

from .. import read_config
from .triage import triage
from .coreset import coreset
from .mask import getmask
from .templates import get_and_merge_templates as gam_templates
from ..mfm import spikesort, suffStatistics, merge_move, cluster_triage
from ..geometry import order_channels_by_distance


def run(score, spike_index_clear, spike_index_collision,
        output_directory='tmp/', recordings_filename='standarized.bin'):
    """Process spikes

    Parameters
    ----------
    score: numpy.ndarray (n_spikes, n_features, n_channels)
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels

    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    spike_index_collision: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for collided spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to generate the templates, defaults to
        whitened.bin

    Returns
    -------
    spike_train_clear: numpy.ndarray (n_clear_spikes, 2)
        A 2D array for clear spikes whose first column indicates the spike
        time and the second column the neuron id determined by the clustering
        algorithm

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    spike_index_collision: numpy.ndarray (n_collided_spikes, 2)
        A 2D array for collided spikes whose first column indicates the spike
        time and the second column the neuron id determined by the clustering
        algorithm

    Examples
    --------

    .. literalinclude:: ../examples/process.py

    """
    CONFIG = read_config()
    MAIN_CHANNEL = 1

    startTime = datetime.datetime.now()

    Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

    logger = logging.getLogger(__name__)

    nG = len(CONFIG.channelGroups)
    nneigh = np.max(np.sum(CONFIG.neighChannels, 0))
    n_coreset = 0
    K = 0

    # first column: spike_time
    # second column: cluster id
    spike_train_clear = np.zeros((0, 2), 'int32')

    if CONFIG.clustering.clustering_method == '2+3':
        spike_index_clear_proc = np.zeros((0, 2), 'int32')
        main_channel_index = spike_index_clear[:, MAIN_CHANNEL]
        for i, c in enumerate(np.unique(main_channel_index)):
            logger.info('Processing channel {}'.format(i))
            idx = main_channel_index == c
            score_c = score[idx]
            spike_index_clear_c = spike_index_clear[idx]

            ##########
            # Triage #
            ##########

            # TODO: refactor this as CONFIG.doTriage was removed
            doTriage = True
            _b = datetime.datetime.now()
            logger.info('Triaging events with main channel {}'.format(c))
            index_keep = triage(score_c, 0, CONFIG.triage.nearest_neighbors,
                                CONFIG.triage.percent, doTriage)
            Time['t'] += (datetime.datetime.now() - _b).total_seconds()

            # add untriaged spike index to spike_index_clear_group
            # and triaged spike index to spike_index_collision
            spike_index_clear_proc = np.concatenate((spike_index_clear_proc,
                                                     spike_index_clear_c[
                                                         index_keep]),
                                                    axis=0)
            spike_index_collision = np.concatenate(
                (spike_index_collision, spike_index_clear_c[~index_keep]),
                axis=0)

            # TODO: add documentation for all of this part, until the
            # "cleaning" commend

            # keep untriaged score only
            score_c = score_c[index_keep]
            group = np.arange(score_c.shape[0])
            mask = np.ones([score_c.shape[0], 1])
            _b = datetime.datetime.now()
            logger.info('Clustering events with main channel {}'.format(c))
            if i == 0:
                global_vbParam, global_maskedData = spikesort(score_c, mask,
                                                              group, CONFIG)
                score_proc = score_c
            else:
                local_vbParam, local_maskedData = spikesort(score_c,
                                                            mask,
                                                            group,
                                                            CONFIG)
                global_vbParam.muhat = np.concatenate([global_vbParam.muhat,
                                                       local_vbParam.muhat],
                                                      axis=1)
                global_vbParam.Vhat = np.concatenate([global_vbParam.Vhat,
                                                      local_vbParam.Vhat],
                                                     axis=2)
                global_vbParam.invVhat = np.concatenate(
                    [global_vbParam.invVhat,
                     local_vbParam.invVhat],
                    axis=2)
                global_vbParam.lambdahat = np.concatenate(
                    [global_vbParam.lambdahat,
                     local_vbParam.lambdahat],
                    axis=0)
                global_vbParam.nuhat = np.concatenate([global_vbParam.nuhat,
                                                       local_vbParam.nuhat],
                                                      axis=0)
                global_vbParam.ahat = np.concatenate([global_vbParam.ahat,
                                                      local_vbParam.ahat],
                                                     axis=0)
                global_maskedData.sumY = np.concatenate(
                    [global_maskedData.sumY,
                     local_maskedData.sumY],
                    axis=0)
                global_maskedData.sumYSq = np.concatenate(
                    [global_maskedData.sumYSq,
                     local_maskedData.sumYSq],
                    axis=0)
                global_maskedData.sumEta = np.concatenate(
                    [global_maskedData.sumEta,
                     local_maskedData.sumEta],
                    axis=0)
                global_maskedData.weight = np.concatenate(
                    [global_maskedData.weight,
                     local_maskedData.weight],
                    axis=0)
                global_maskedData.groupMask = np.concatenate(
                    [global_maskedData.groupMask,
                     local_maskedData.groupMask],
                    axis=0)
                global_maskedData.meanY = np.concatenate(
                    [global_maskedData.meanY,
                     local_maskedData.meanY],
                    axis=0)
                global_maskedData.meanYSq = np.concatenate(
                    [global_maskedData.meanYSq,
                     local_maskedData.meanYSq],
                    axis=0)
                global_maskedData.meanEta = np.concatenate(
                    [global_maskedData.meanEta,
                     local_maskedData.meanEta],
                    axis=0)
                score_proc = np.concatenate([score_proc, score_c], axis=0)

        logger.info('merging all channels')
        L = np.ones(global_vbParam.muhat.shape[1])
        global_vbParam.update_local(global_maskedData)
        suffStat = suffStatistics(global_maskedData, global_vbParam)
        global_vbParam, suffStat, L = merge_move(
            global_maskedData, global_vbParam, suffStat, CONFIG, L, 0)
        assignmentTemp = np.argmax(global_vbParam.rhat, axis=1)
        assignment = np.zeros(score_proc.shape[0], 'int16')

        for j in range(score_proc.shape[0]):
            assignment[j] = assignmentTemp[j]

        idx_triage = cluster_triage(global_vbParam, score_proc, 3)
        assignment[idx_triage] = -1
        Time['s'] += (datetime.datetime.now() - _b).total_seconds()

        ############
        # Cleaning #
        ############

        # TODO: describe this step

        spike_train_clear = np.concatenate(
            [
                spike_index_clear_proc[~idx_triage, 0:1:],
                assignment[~idx_triage, np.newaxis]
            ],
            axis=1)
        spike_index_collision = np.concatenate(
            [spike_index_collision, spike_index_clear_proc[idx_triage]])

    else:

        # according to the docs if clustering method is not 2+3, you can set
        # 3 x neighboring_channels, but I do not see where the
        # neighboring_channels is being parsed on this else statemente

        c_idx = np.ones((CONFIG.recordings.n_channels,
                         nneigh), 'int32') * CONFIG.recordings.n_channels

        for c in range(CONFIG.recordings.n_channels):
            ch_idx, _ = order_channels_by_distance(
                c,
                np.where(CONFIG.neighChannels[c])[0], CONFIG.geom)
            c_idx[c, :ch_idx.shape[0]] = ch_idx

        # iterate over every channel group [missing documentation for this
        # function]. why is this order needed?
        for g in range(nG):
            logger.info("Processing group {} in {} groups.".format(g + 1, nG))
            logger.info("Processiing data (triage, coreset, masking) ...")
            channels = CONFIG.channelGroups[g]
            neigh_chans = np.where(
                np.sum(CONFIG.neighChannels[channels], axis=0) > 0)[0]

            score_group = np.zeros((0, CONFIG.spikes.temporal_features,
                                    neigh_chans.shape[0]))
            coreset_id_group = np.zeros((0), 'int32')
            mask_group = np.zeros((0, neigh_chans.shape[0]))
            spike_index_clear_group = np.zeros((0, 2), 'int32')

            # go through every channel in the group
            for c in channels:

                # index of data whose main channel is c
                idx = spike_index_clear[:, MAIN_CHANNEL] == c
                if np.sum(idx) > 0:

                    # score whose main channel is c
                    score_c = score[idx]
                    # spike_index_clear whose main channel is c
                    spike_index_clear_c = spike_index_clear[idx]

                    ##########
                    # Triage #
                    ##########

                    # TODO: refactor this as CONFIG.doTriage was removed
                    doTriage = True

                    _b = datetime.datetime.now()
                    index_keep = triage(score_c, 0,
                                        CONFIG.triage.nearest_neighbors,
                                        CONFIG.triage.percent, doTriage)
                    Time['t'] += (datetime.datetime.now() - _b).total_seconds()

                    # add untriaged spike index to spike_index_clear_group
                    # and triaged spike index to spike_index_collision
                    spike_index_clear_group = np.concatenate(
                        (spike_index_clear_group,
                         spike_index_clear_c[index_keep]),
                        axis=0)
                    spike_index_collision = np.concatenate(
                        (spike_index_collision,
                         spike_index_clear_c[~index_keep]),
                        axis=0)

                    # keep untriaged score only
                    score_c = score_c[index_keep]

                    ###########
                    # Coreset #
                    ###########

                    # TODO: refactor this as CONFIG.doCoreset was removed
                    doCoreset = True

                    _b = datetime.datetime.now()
                    coreset_id = coreset(score_c, CONFIG.coreset.clusters,
                                         CONFIG.coreset.threshold, doCoreset)
                    Time['c'] += (datetime.datetime.now() - _b).total_seconds()

                    ###########
                    # Masking #
                    ###########

                    _b = datetime.datetime.now()
                    mask = getmask(score_c, coreset_id,
                                   CONFIG.clustering.masking_threshold,
                                   CONFIG.spikes.temporal_features)
                    Time['m'] += (datetime.datetime.now() - _b).total_seconds()

                    ################
                    # collect data #
                    ################

                    # restructure score_c and mask to have same number of
                    # channels as score_group
                    score_temp = np.zeros(
                        (score_c.shape[0], CONFIG.spikes.temporal_features,
                         neigh_chans.shape[0]))
                    mask_temp = np.zeros((mask.shape[0], neigh_chans.shape[0]))
                    nneigh_c = np.sum(c_idx[c] < CONFIG.recordings.n_channels)
                    for j in range(nneigh_c):
                        c_interest = np.where(neigh_chans == c_idx[c, j])[0][0]
                        score_temp[:, :, c_interest] = score_c[:, :, j]
                        mask_temp[:, c_interest] = mask[:, j]

                    # add score, coreset_id, mask to the groups
                    score_group = np.concatenate(
                        (score_group, score_temp), axis=0)
                    mask_group = np.concatenate(
                        (mask_group, mask_temp), axis=0)
                    coreset_id_group = np.concatenate(
                        (coreset_id_group, coreset_id + n_coreset + 1), axis=0)
                    n_coreset += np.max(coreset_id) + 1

            if score_group.shape[0] > 0:
                ##############
                # Clustering #
                ##############

                _b = datetime.datetime.now()
                logger.info("Clustering...")
                coreset_id_group = coreset_id_group - 1
                n_coreset = 0
                cluster_id = spikesort(score_group, mask_group,
                                       coreset_id_group, CONFIG)
                Time['s'] += (datetime.datetime.now() - _b).total_seconds()

                ############
                # Cleaning #
                ############

                # model based triage
                idx_triage = (cluster_id == -1)

                # concatenate spike index with cluster id of untriaged ones
                # to create spike_train_clear
                si_clustered = spike_index_clear_group[~idx_triage]
                spt = si_clustered[:, [0]]
                cluster_id = cluster_id[~idx_triage][:, np.newaxis]

                spike_train_temp = np.concatenate(
                    (spt, cluster_id + K), axis=1)
                spike_train_clear = np.concatenate(
                    (spike_train_clear, spike_train_temp), axis=0)
                K += np.amax(cluster_id) + 1

                # concatenate triaged spike_index_clear_group
                # into spike_index_collision
                spike_index_collision = np.concatenate(
                    (spike_index_collision,
                     spike_index_clear_group[idx_triage]),
                    axis=0)

    #################
    # Get templates #
    #################

    _b = datetime.datetime.now()
    logger.info("Getting Templates...")
    path_to_recordings = os.path.join(CONFIG.data.root_folder,
                                      output_directory, recordings_filename)
    merge_threshold = CONFIG.templates.merge_threshold
    spike_train_clear, templates = gam_templates(
        spike_train_clear, path_to_recordings, CONFIG.spikeSize,
        CONFIG.templatesMaxShift, merge_threshold, CONFIG.neighChannels)

    Time['e'] += (datetime.datetime.now() - _b).total_seconds()

    currentTime = datetime.datetime.now()

    if CONFIG.clustering.clustering_method == '2+3':
        logger.info("Mainprocess done in {0} seconds.".format(
            (currentTime - startTime).seconds))
        logger.info("\ttriage:\t{0} seconds".format(Time['t']))
        logger.info("\tclustering:\t{0} seconds".format(Time['s']))
        logger.info("\ttemplates:\t{0} seconds".format(Time['e']))
    else:
        logger.info("\ttriage:\t{0} seconds".format(Time['t']))
        logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
        logger.info("\tmasking:\t{0} seconds".format(Time['m']))
        logger.info("\tclustering:\t{0} seconds".format(Time['s']))
        logger.info("\ttemplates:\t{0} seconds".format(Time['e']))

    return spike_train_clear, templates, spike_index_collision
