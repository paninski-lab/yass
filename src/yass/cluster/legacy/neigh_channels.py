from os.path import join
import logging
import datetime
import numpy as np

from yass import read_config
from yass.cluster.legacy.subsample import random_subsample
from yass.cluster.legacy.triage import triage
from yass.cluster.legacy.coreset import coreset
from yass.cluster.legacy.mask import getmask
from yass.cluster.legacy.util import (run_cluster, calculate_sparse_rhat)
from yass.mfm import get_core_data


def neigh_channels(spike_index):
    """neigh channels clustering

    Parameters
    ----------
    spike_index: numpy.ndarray (n_clear_spikes, 2), str or Path
        2D array with indexes for spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum). Or path to an npy file

    Returns
    -------
    spike_train
    """
    CONFIG = read_config()

    # load files in case they are strings or Path objects
    path_to_scores = join(CONFIG.path_to_output_directory,
                          'detect', 'scores_clear.npy')
    scores = np.load(path_to_scores)

    startTime = datetime.datetime.now()

    scores_all = np.copy(scores)
    spike_index_all = np.copy(spike_index)

    Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

    logger = logging.getLogger(__name__)

    ##########
    # Triage #
    ##########

    _b = datetime.datetime.now()
    logger.info("Randomly subsampling...")
    scores, spike_index = random_subsample(scores, spike_index,
                                           CONFIG.cluster.max_n_spikes)
    logger.info("Triaging...")
    scores, spike_index = triage(scores, spike_index,
                                 CONFIG.cluster.triage.nearest_neighbors,
                                 CONFIG.cluster.triage.percent,
                                 location_feature=False)
    Time['t'] += (datetime.datetime.now()-_b).total_seconds()

    ###########
    # Coreset #
    ###########
    _b = datetime.datetime.now()
    logger.info("Coresetting...")
    groups = coreset(scores,
                     spike_index,
                     CONFIG.cluster.coreset.clusters,
                     CONFIG.cluster.coreset.threshold)
    Time['c'] += (datetime.datetime.now() - _b).total_seconds()

    ###########
    # Masking #
    ###########
    _b = datetime.datetime.now()
    logger.info("Masking...")
    masks = getmask(scores, spike_index, groups,
                    CONFIG.cluster.masking_threshold)
    Time['m'] += (datetime.datetime.now() - _b).total_seconds()

    ##############
    # Clustering #
    ##############
    _b = datetime.datetime.now()
    logger.info("Clustering...")
    vbParam, tmp_loc, scores, spike_index = run_cluster(
        scores, masks, groups, spike_index,
        CONFIG.cluster.min_spikes, CONFIG)
    Time['s'] += (datetime.datetime.now()-_b).total_seconds()

    vbParam.rhat = calculate_sparse_rhat(vbParam, tmp_loc, scores_all,
                                         spike_index_all,
                                         CONFIG.neigh_channels)
    idx_keep = get_core_data(vbParam, scores_all, np.inf, 2)
    spike_train = vbParam.rhat[idx_keep]
    spike_train[:, 0] = spike_index_all[spike_train[:, 0].astype('int32'), 0]

    # report timing
    currentTime = datetime.datetime.now()
    logger.info("Mainprocess done in {0} seconds.".format(
        (currentTime - startTime).seconds))
    logger.info("\ttriage:\t{0} seconds".format(Time['t']))
    logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
    logger.info("\tmasking:\t{0} seconds".format(Time['m']))
    logger.info("\tclustering:\t{0} seconds".format(Time['s']))

    return spike_train, tmp_loc, vbParam
