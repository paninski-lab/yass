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
from .cluster import runSorter
from .clean import clean_output
from .templates import get_templates
from ..MFM import spikesort
from ..geometry import order_channels_by_distance


def run(score, spike_index_clear, spike_index_collision):
    """Run process pipeline

    Parameters
    ----------
    score: list
        As returned from the preprocessor
    clear_index: list
        As returned from the preprocessor
    spike_times: list
        As returned form the preprocessor

    Returns
    -------
    spike_train:
        A (number of spikes x 2) matrix whose first column indicates whe
        spike time and the second column the neuron id, where the neuron id
        is determined by the clustering algorithm

    spike_times_left: list
        A list of length n_chanels whose first column indicates the spike
        time for a potential spike

    templates: numpy.ndarray
        A (number of channels x waveform size x number of templates)
        multidimensional array containing the templates obtained

    Examples
    --------

    .. literalinclude:: ../examples/process.py
    """
    CONFIG = read_config()
    SPIKE_TIME, MAIN_CHANNEL = 0, 1

    startTime = datetime.datetime.now()

    Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

    logger = logging.getLogger(__name__)

    nG = len(CONFIG.channelGroups)
    nneigh = np.max(np.sum(CONFIG.neighChannels, 0))
    n_coreset = 0
    K = 0

    # first column: spike_time
    # second column: cluster id
    spike_train_clear = np.zeros((0,2), 'int32')

    # order of channels
    c_idx = np.ones((CONFIG.recordings.n_channels, nneigh), 'int32')*CONFIG.recordings.n_channels
    for c in range(CONFIG.recordings.n_channels):
        ch_idx, _ = order_channels_by_distance(c,
                                               np.where(
                                               CONFIG.neighChannels[c])[0],
                                               CONFIG.geom)
        c_idx[c,:ch_idx.shape[0]] = ch_idx


    for g in range(nG):
        logger.info("Processing group {} in {} groups.".format(g+1, nG))
        logger.info("Processiing data (triage, coreset, masking) ...")
        channels = CONFIG.channelGroups[g]
        neigh_chans = np.where(np.sum(CONFIG.neighChannels[channels], axis=0) > 0)[0]

        score_group = np.zeros((0, CONFIG.spikes.temporal_features, neigh_chans.shape[0]))
        coreset_id_group = np.zeros((0), 'int32')
        mask_group = np.zeros((0, neigh_chans.shape[0]))
        spike_index_clear_group = np.zeros((0, 2), 'int32')
        for c in channels:

            # index of data whose main channel is c
            idx = spike_index_clear[:,MAIN_CHANNEL] == c
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
                index_keep = triage(score_c, 0, CONFIG.triage.nearest_neighbors,
                                    CONFIG.triage.percent, doTriage)
                Time['t'] += (datetime.datetime.now()-_b).total_seconds()

                # add untriaged spike index to spike_index_clear_group
                # and triaged spike index to spike_index_collision
                spike_index_clear_group = np.concatenate((
                    spike_index_clear_group, spike_index_clear_c[index_keep]),
                    axis = 0)
                spike_index_collision = np.concatenate((
                    spike_index_collision, spike_index_clear_c[~index_keep]),
                    axis = 0)

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
                Time['c'] += (datetime.datetime.now()-_b).total_seconds()


                ###########
                # Masking #
                ###########

                _b = datetime.datetime.now()
                mask = getmask(score_c, coreset_id, CONFIG.clustering.masking_threshold, CONFIG.spikes.temporal_features)
                Time['m'] += (datetime.datetime.now()-_b).total_seconds()


                ################
                # collect data #
                ################

                # restructure score_c and mask to have same number of channels
                # as score_group
                score_temp = np.zeros((score_c.shape[0],
                    CONFIG.spikes.temporal_features, neigh_chans.shape[0]))
                mask_temp = np.zeros((mask.shape[0],neigh_chans.shape[0]))
                nneigh_c = np.sum(c_idx[c] < CONFIG.recordings.n_channels)
                for j in range(nneigh_c):
                    c_interest = np.where(neigh_chans == c_idx[c,j])[0][0]
                    score_temp[:,:,c_interest] = score_c[:,:,j]
                    mask_temp[:,c_interest] = mask[:,j]

                # add score, coreset_id, mask to the groups
                score_group = np.concatenate(
                    (score_group, score_temp), axis = 0)
                mask_group = np.concatenate(
                    (mask_group, mask_temp), axis = 0)
                coreset_id_group = np.concatenate(
                    (coreset_id_group, coreset_id + n_coreset + 1), axis = 0)
                n_coreset += np.max(coreset_id) + 1


        if score_group.shape[0] > 0:

            ##############
            # Clustering #
            ##############

            _b = datetime.datetime.now()
            logger.info("Clustering...")
            coreset_id_group = coreset_id_group - 1
            n_coreset = 0
            cluster_id = spikesort(score_group, mask_group, coreset_id_group, CONFIG)
            Time['s'] += (datetime.datetime.now()-_b).total_seconds()


            ############
            # Cleaning #
            ############

            # model based triage
            idx_triage = (cluster_id == -1)

            # concatenate spike index with cluster id of untriaged ones
            # to create spike_train_clear
            si_clustered = spike_index_clear_group[~idx_triage]
            spt = si_clustered[:,[0]]
            cluster_id = cluster_id[~idx_triage][:, np.newaxis]

            spike_train_temp = np.concatenate((spt,cluster_id+K), axis=1)
            spike_train_clear = np.concatenate(
                (spike_train_clear, spike_train_temp), axis=0)
            K += np.amax(cluster_id)+1

            # concatenate triaged spike_index_clear_group
            # into spike_index_collision
            spike_index_collision = np.concatenate((
                spike_index_collision, spike_index_clear_group[idx_triage]),
                axis = 0)

    #################
    # Get templates #
    #################

    _b = datetime.datetime.now()
    logger.info("Getting Templates...")
    path_to_whiten = os.path.join(CONFIG.data.root_folder, 'tmp/whiten.bin')
    spike_train_clear, templates = get_templates(spike_train_clear,
                                                 CONFIG.batch_size,
                                                 CONFIG.BUFF,
                                                 CONFIG.nBatches,
                                                 CONFIG.recordings.n_channels,
                                                 CONFIG.spikeSize,
                                                 CONFIG.templatesMaxShift,
                                                 CONFIG.scaleToSave,
                                                 CONFIG.neighChannels,
                                                 path_to_whiten,
                                                 CONFIG.templates.merge_threshold)
    Time['e'] += (datetime.datetime.now()-_b).total_seconds()


    currentTime = datetime.datetime.now()

    logger.info("Mainprocess done in {0} seconds.".format(
        (currentTime-startTime).seconds))
    logger.info("\ttriage:\t{0} seconds".format(Time['t']))
    logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
    logger.info("\tmasking:\t{0} seconds".format(Time['m']))
    logger.info("\tclustering:\t{0} seconds".format(Time['s']))
    logger.info("\ttemplates:\t{0} seconds".format(Time['e']))

    return spike_train_clear, templates, spike_index_collision
