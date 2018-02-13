"""
Process pipeline: triage, coreset, masking, clustering,
getting templates and cleaning
"""
import os
import logging
import datetime

from yass import read_config
from yass.process.list import make_list
from yass.process.subsample import random_subsample
from yass.process.triage import triage
from yass.process.coreset import coreset
from yass.process.mask import getmask
from yass.process.cluster import run_cluster, run_cluster_loccation
from yass.process.templates import get_and_merge_templates as gam_templates


def run(scores, spike_index,
        output_directory='tmp/', recordings_filename='standarized.bin'):
    """Process spikes

    Parameters
    ----------
    score: numpy.ndarray (n_spikes, n_features, n_channels)
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels

    spike_index: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for spikes, first column contains the
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

    Examples
    --------

    .. literalinclude:: ../examples/process.py

    """
    CONFIG = read_config()

    startTime = datetime.datetime.now()

    Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

    logger = logging.getLogger(__name__)

    # transform data structure of scores and spike_index
    scores, spike_index = make_list(scores, spike_index,
                                    CONFIG.recordings.n_channels)

    ##########
    # Triage #
    ##########
    _b = datetime.datetime.now()
    logger.info("Triaging...")
    score, spike_index = triage(scores, spike_index,
                                CONFIG.triage.nearest_neighbors,
                                CONFIG.triage.percent)

    logger.info("Randomly subsampling...")
    scores, spike_index = random_subsample(scores, spike_index,
                                           CONFIG.max_n_spikes)
    Time['t'] += (datetime.datetime.now()-_b).total_seconds()

    if CONFIG.clustering.clustering_method == 'location':
        ##############
        # Clustering #
        ##############
        _b = datetime.datetime.now()
        logger.info("Clustering...")
        spike_train = run_cluster_loccation(scores,
                                            spike_index, CONFIG)
        Time['s'] += (datetime.datetime.now()-_b).total_seconds()

    else:
        ###########
        # Coreset #
        ###########
        _b = datetime.datetime.now()
        logger.info("Coresetting...")
        groups = coreset(scores,
                         CONFIG.coreset.clusters,
                         CONFIG.coreset.threshold)
        Time['c'] += (datetime.datetime.now() - _b).total_seconds()

        ###########
        # Masking #
        ###########
        _b = datetime.datetime.now()
        logger.info("Masking...")
        masks = getmask(scores, groups,
                        CONFIG.clustering.masking_threshold)
        Time['m'] += (datetime.datetime.now() - _b).total_seconds()

        ##############
        # Clustering #
        ##############
        _b = datetime.datetime.now()
        logger.info("Clustering...")
        spike_train = run_cluster(scores, masks, groups,
                                  spike_index, CONFIG)
        Time['s'] += (datetime.datetime.now()-_b).total_seconds()

    #################
    # Get templates #
    #################
    _b = datetime.datetime.now()
    logger.info("Getting Templates...")
    path_to_recordings = os.path.join(CONFIG.data.root_folder,
                                      output_directory, recordings_filename)
    merge_threshold = CONFIG.templates.merge_threshold
    spike_train, templates = gam_templates(
        spike_train, path_to_recordings, CONFIG.spikeSize,
        CONFIG.templatesMaxShift, merge_threshold, CONFIG.neighChannels)
    Time['e'] += (datetime.datetime.now() - _b).total_seconds()

    # report timing
    currentTime = datetime.datetime.now()
    logger.info("Mainprocess done in {0} seconds.".format(
        (currentTime - startTime).seconds))
    logger.info("\ttriage:\t{0} seconds".format(Time['t']))
    logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
    logger.info("\tmasking:\t{0} seconds".format(Time['m']))
    logger.info("\tclustering:\t{0} seconds".format(Time['s']))
    logger.info("\ttemplates:\t{0} seconds".format(Time['e']))

    return spike_train, templates
