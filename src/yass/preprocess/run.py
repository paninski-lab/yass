"""
Preprocess pipeline
"""
import datetime
import logging
import os.path

import numpy as np

from .. import read_config
from ..batch import BatchProcessorFactory

from .detect import threshold_detection
from .filter import whitening_matrix, whitening, butterworth
from .score import get_score_pca, get_pca_suff_stat, get_pca_projection
from .waveform import get_waveforms
from .standarize import standarize, sd
from ..neuralnet import NeuralNetDetector, NeuralNetTriage

# remove this
Q = None


def run():
    """Execute preprocessing pipeline

    Returns
    -------
    score: list
        List of size n_channels, each list contains a (clear spikes x
        number of features x number of channels) multidimensional array
        score for every clear spike

    clear_index: list
        List of size n_channels, each list contains the indexes in
        spike_times (first column) where the spike was clear

    spike_times: list
        List with n_channels elements, each element contains spike times
        in the first column and [SECOND COLUMN?]

    Examples
    --------

    .. literalinclude:: ../examples/preprocess.py
    """
    logger = logging.getLogger(__name__)

    start_time = datetime.datetime.now()
    time = {'f': 0, 's': 0, 'd': 0, 'w': 0, 'b': 0, 'e': 0}

    # FIXME: remove this
    CONFIG = read_config()
    whiten_file = open(os.path.join(CONFIG.root, 'tmp/whiten.bin'), 'wb')

    # initialize processor for raw data
    path = os.path.join(CONFIG.root, CONFIG.filename)
    dtype = CONFIG.dtype

    # initialize factory
    factory = BatchProcessorFactory(path_to_file=None,
                                    dtype=None,
                                    n_channels=CONFIG.nChan,
                                    max_memory=CONFIG.maxMem,
                                    buffer_size=None)

    if CONFIG.doFilter == 1:

        _b = datetime.datetime.now()
        # make batch processor for raw data -> buterworth -> filtered
        bp = factory.make(path_to_file=path, dtype=dtype,
                          buffer_size=0)
        logger.info('Initialized butterworth batch processor: {}'
                    .format(bp))

        # run filtering
        path = os.path.join(CONFIG.root,  'tmp/filtered.bin')
        dtype = bp.process_function(butterworth,
                                    path,
                                    CONFIG.filterLow,
                                    CONFIG.filterHighFactor,
                                    CONFIG.filterOrder,
                                    CONFIG.srate)
        time['f'] += (datetime.datetime.now()-_b).total_seconds()

    # TODO: cache computations
    # make batch processor for filtered -> standarize -> standarized
    _b = datetime.datetime.now()
    bp = factory.make(path_to_file=path, dtype=dtype, buffer_size=0)

    # compute the standard deviation using the first batch only
    batch1 = next(bp)
    sd_ = sd(batch1, CONFIG.srate)

    # make another batch processor
    bp = factory.make(path_to_file=path, dtype=dtype, buffer_size=0)
    logger.info('Initialized standarization batch processor: {}'
                .format(bp))

    # run standarization
    path = os.path.join(CONFIG.root,  'tmp/standarized.bin')
    dtype = bp.process_function(standarize,
                                path,
                                sd_)
    time['s'] += (datetime.datetime.now()-_b).total_seconds()

    # create another batch processor for the rest of the pipeline
    bp = factory.make(path_to_file=path, dtype=dtype,
                      buffer_size=CONFIG.BUFF)
    logger.info('Initialized preprocess batch processor: {}'
                .format(bp))

    # initialize output variables
    get_score = 1
    spike_index_clear = None
    spike_index_collision = None
    score = None
    pca_suff_stat = None
    spikes_per_channel = None

    for i, batch in enumerate(bp):

        # load nueral net detector if necessary:
        if CONFIG.detctionMethod == 'nn':
            nnDetector = NeuralNetDetector(CONFIG)
            proj = nnDetector.load_w_ae()
            nnTriage = NeuralNetTriage(CONFIG)
        else:
            nnDetector = None
            proj = None
            nnTriage = None

        if i > CONFIG.nPortion:
            get_score = 0

        # process batch
        # spike index is defined as a location in each minibatch
        (si_clr_batch, score_batch, si_col_batch,
         pss_batch, spc_batch,
         time) = process_batch(batch, get_score, CONFIG.BUFF, time,
                               nnDetector=nnDetector, proj=proj,
                               nnTriage=nnTriage, whiten_file=whiten_file)

        # add batch number to spike_index
        batch_ids = np.ones((si_clr_batch.shape[0], 1), 'int32') * i
        si_clr_batch = np.hstack((si_clr_batch, batch_ids))

        batch_ids = np.ones((si_col_batch.shape[0], 1), 'int32') * i
        si_col_batch = np.hstack((si_col_batch, batch_ids))

        if i == 0:
            spike_index_clear = si_clr_batch
            spike_index_collision = si_col_batch
            score = score_batch

            pca_suff_stat = pss_batch
            spikes_per_channel = spc_batch

        else:
            spike_index_clear = np.vstack((spike_index_clear,
                si_clr_batch))
            spike_index_collision = np.vstack((spike_index_collision,
                si_col_batch))
            if get_score == 1:
                score = np.concatenate((score, score_batch), axis = 0)
            pca_suff_stat += pss_batch
            spikes_per_channel += spc_batch

    whiten_file.close()

    if CONFIG.detctionMethod != 'nn':
        _b = datetime.datetime.now()
        rot = get_pca_projection(pca_suff_stat, spikes_per_channel,
                                 CONFIG.nFeat, CONFIG.neighChannels)
        score = get_score_pca(spike_index_clear, rot, CONFIG.neighChannels,
                              CONFIG.geom, CONFIG.batch_size + 2*CONFIG.BUFF,
                              os.path.join(CONFIG.root,'tmp/whiten.bin'),
                              CONFIG.scaleToSave)

        time['e'] += (datetime.datetime.now()-_b).total_seconds()

    # timing
    current_time = datetime.datetime.now()
    logger.info("Preprocessing done in {0} seconds.".format(
                     (current_time-start_time).seconds))
    logger.info("\tfiltering:\t{0} seconds".format(time['f']))
    logger.info("\tstandardization:\t{0} seconds".format(time['s']))
    logger.info("\tdetection:\t{0} seconds".format(time['d']))
    logger.info("\twhitening:\t{0} seconds".format(time['w']))
    logger.info("\tsaving recording:\t{0} seconds".format(time['b']))
    logger.info("\tgetting waveforms:\t{0} seconds".format(time['e']))

    return score, spike_index_clear, spike_index_collision


def process_batch(rec, get_score, BUFF, time, nnDetector, proj, nnTriage,
                  whiten_file):
    logger = logging.getLogger(__name__)
    CONFIG = read_config()

    # detect spikes
    _b = datetime.datetime.now()
    logger.info('running detection')
    if CONFIG.detctionMethod == 'nn':
        spike_index = nnDetector.get_spikes(rec)

    else:
        spike_index = threshold_detection(rec,
                                    CONFIG.neighChannels,
                                    CONFIG.spikeSize,
                                    CONFIG.stdFactor)

    # From Peter: When the recording is too long, I load them by
    # little chunk by chunk (chunk it time-wise). But I also add
    # some buffer. If the detected spike time is in the buffer,
    # i remove that because it will be detected in another chunk
    spike_index = spike_index[np.logical_and(spike_index[:, 0] > BUFF,
                  spike_index[:, 0] < (rec.shape[0] - BUFF))]

    time['d'] += (datetime.datetime.now()-_b).total_seconds()

    # get withening matrix per batch or onece in total
    if CONFIG.doWhitening == 1:
        _b = datetime.datetime.now()

        global Q
        if CONFIG.whitenBatchwise or Q is None:
            # cache this
            Q = whitening_matrix(rec, CONFIG.neighChannels,
                                 CONFIG.spikeSize)

        rec = whitening(rec, Q)

        time['w'] += (datetime.datetime.now()-_b).total_seconds()

    _b = datetime.datetime.now()

    # save whiten data
    chunk = rec*CONFIG.scaleToSave
    chunk.reshape(chunk.shape[0]*chunk.shape[1])
    chunk.astype('int16').tofile(whiten_file)

    time['b'] += (datetime.datetime.now()-_b).total_seconds()

    
    _b = datetime.datetime.now()
    if get_score == 0:
        # if we are not calculating score for this minibatch, every spikes
        # are considered as collision and will be referred during deconvlution
        spike_index_clear = np.zeros((0,2), 'int32')
        score = None
        spike_index_collision = spike_index

        pca_suff_stat = 0
        spikes_per_channel = 0

    elif CONFIG.detctionMethod == 'nn':
        # with nn, get scores and triage bad ones
        (spike_index_clear, score,
        spike_index_collision) = get_waveforms(rec,
                                               spike_index,
                                               proj,
                                               CONFIG.neighChannels,
                                               CONFIG.geom,
                                               nnTriage,
                                               CONFIG.nnThreshdoldCol)

        # since we alread have scores, no need to calculated sufficient
        # statistics for pca
        pca_suff_stat = 0
        spikes_per_channel = 0

    elif CONFIG.detctionMethod == 'threshold':
        # every spikes are considered as clear spikes as no triage is done
        spike_index_clear = spike_index
        score = None
        spike_index_collision = np.zeros((0,2), 'int32')

        # get sufficient statistics for pca if we don't have projection matrix
        pca_suff_stat, spikes_per_channel = get_pca_suff_stat(rec, spike_index,
            CONFIG.spikeSize)

    time['e'] += (datetime.datetime.now()-_b).total_seconds()

    return (spike_index_clear, score, spike_index_collision,
        pca_suff_stat, spikes_per_channel, time)
