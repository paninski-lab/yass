"""
Process pipeline: triage (optional) coreset (optional), masking, clustering,
getting templates and cleaning
"""
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


def run(score, clear_index, spike_times):
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
        spike time and the second column the neuron id
    spike_times_left: list
        A list of length n_chanels whose first column indicates the spike
        time for a potential spike, [SECOND COLUMN?]
    templates:
        ?

    Examples
    --------

    .. literalinclude:: ../examples/process.py
    """
    CONFIG = read_config()

    startTime = datetime.datetime.now()

    Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

    logger = logging.getLogger(__name__)

    ##########
    # Triage #
    ##########

    if CONFIG.doTriage:
        _b = datetime.datetime.now()
        logger.info("Triaging...")
        score, clear_index = triage(score, clear_index, CONFIG.nChan,
                                    CONFIG.triageK, CONFIG.triagePercent,
                                    CONFIG.neighChannels, CONFIG.doTriage)
        Time['t'] += (datetime.datetime.now()-_b).total_seconds()

    # FIXME: pipeline will fail if coreset is deactivate as making is using
    # coreset output as input

    ###########
    # Coreset #
    ###########

    if CONFIG.doCoreset:
        _b = datetime.datetime.now()
        logger.info("Coresettting...")
        group = coreset(score, CONFIG.nChan, CONFIG.coresetK, CONFIG.coresetTh)
        Time['c'] += (datetime.datetime.now()-_b).total_seconds()

    ###########
    # Masking #
    ###########

    _b = datetime.datetime.now()
    logger.info("Masking...")
    mask = getmask(score, group, CONFIG.maskTh, CONFIG.nFeat, CONFIG.nChan,
                   CONFIG.doCoreset)
    Time['m'] += (datetime.datetime.now()-_b).total_seconds()

    ##############
    # Clustering #
    ##############

    _b = datetime.datetime.now()
    logger.info("Clustering...")
    spike_train_clear = runSorter(score, mask, clear_index, group,
                                  CONFIG.channelGroups, CONFIG.neighChannels,
                                  CONFIG.nFeat, CONFIG)
    Time['s'] += (datetime.datetime.now()-_b).total_seconds()

    #################
    # Clean output  #
    #################
    spike_train_clear, spike_times_left = clean_output(spike_train_clear,
                                                 spike_times, clear_index,
                                                 CONFIG.batch_size,
                                                 CONFIG.BUFF)

    #################
    # Get templates #
    #################

    _b = datetime.datetime.now()
    logger.info("Getting Templates...")
    path_to_whiten = os.path.join(CONFIG.root, 'tmp/whiten.bin')
    spike_train, templates = get_templates(spike_train_clear,
                                                 CONFIG.batch_size,
                                                 CONFIG.BUFF,
                                                 CONFIG.nBatches,
                                                 CONFIG.nChan,
                                                 CONFIG.spikeSize,
                                                 CONFIG.templatesMaxShift,
                                                 CONFIG.scaleToSave,
                                                 CONFIG.neighChannels,
                                                 path_to_whiten,
                                                 CONFIG.tMergeTh)
    Time['e'] += (datetime.datetime.now()-_b).total_seconds()


    currentTime = datetime.datetime.now()

    logger.info("Mainprocess done in {0} seconds.".format(
        (currentTime-startTime).seconds))
    logger.info("\ttriage:\t{0} seconds".format(Time['t']))
    logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
    logger.info("\tmasking:\t{0} seconds".format(Time['m']))
    logger.info("\tclustering:\t{0} seconds".format(Time['s']))
    logger.info("\tmake templates:\t{0} seconds".format(Time['e']))

    return spike_train, spike_times_left, templates
