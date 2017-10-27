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
from .score import getPCAProjection, getPcaSS, getScore
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

    # TODO: cache computations
    # make batch processor for filtered -> standarize -> standarized
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

    # create another batch processor for the rest of the pipeline
    bp = factory.make(path_to_file=path, dtype=dtype,
                      buffer_size=CONFIG.BUFF)
    logger.info('Initialized preprocess batch processor: {}'
                .format(bp))

    # bar = progressbar.ProgressBar(maxval=nBatches)
    score = 0
    get_score = 1

    # we will create a (spike times, 2) matrix for every channel, the first
    # column is the spike time and the second column the batch were the spike
    # was processed
    spike_times = [np.empty((0, 2), 'int32')] * CONFIG.nChan

    for i, batch in enumerate(bp):
        start_time = datetime.datetime.now()
        time = {'r': 0, 'f': 0, 's': 0, 'd': 0, 'w': 0, 'b': 0, 'e': 0}

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
        (score_batch, clear_index_batch,
         spike_time_batch, ss_batch,
         n_spikes_batch, time) = process_batch(batch, get_score, CONFIG.BUFF,
                                               time, nnDetector=nnDetector,
                                               proj=proj, nnTriage=nnTriage,
                                               whiten_file=whiten_file)

        # process spike times
        for c in range(CONFIG.nChan):
            # convert list of spike times to a (n, 1) ndarray
            spike_time_arr = spike_time_batch[c][:, np.newaxis]
            # create a (n, i) ndarray of ones
            ones = np.ones((len(spike_time_arr), 1), 'int32') * i
            # replace the value by a (n, 1) ndarray
            spike_time_and_batch_matrix = np.hstack((spike_time_arr, ones))
            # append it to the spike times list in the corresponding element
            spike_times[c] = np.vstack((spike_times[c],
                                        spike_time_and_batch_matrix))

        # TODO: what's going on here?
        if score == 0:
            score = score_batch
            clr_idx = clear_index_batch
            # spt = spike_time_batch
            ss = ss_batch
            nspikes = n_spikes_batch
        else:
            ss += ss_batch
            nspikes += n_spikes_batch

        # bar.update(i+1)

    # bar.finish()
    whiten_file.close()

    # TODO: ask peter, why are we only running this for threshold detector?
    if CONFIG.detctionMethod != 'nn':
        _b = datetime.datetime.now()
        rot = getPCAProjection(ss, nspikes, CONFIG.nFeat,
                               CONFIG.neighChannels)
        score, clr_idx = getScore(spike_times, rot, CONFIG.nChan,
                                  CONFIG.spikeSize,
                                  CONFIG.nFeat,
                                  CONFIG.neighChannels,
                                  os.path.join(CONFIG.root,
                                               'tmp/whiten.bin'),
                                  CONFIG.scaleToSave,
                                  CONFIG.nBatches,
                                  CONFIG.nPortion,
                                  CONFIG.BUFF,
                                  CONFIG.batch_size)
        time['e'] += (datetime.datetime.now()-_b).total_seconds()

    # timing
    current_time = datetime.datetime.now()
    logger.info("Preprocessing done in {0} seconds.".format(
                     (current_time-start_time).seconds))
    logger.info("\treading data:\t{0} seconds".format(time['r']))
    logger.info("\tfiltering:\t{0} seconds".format(time['f']))
    logger.info("\tstandardization:\t{0} seconds".format(time['s']))
    logger.info("\tdetection:\t{0} seconds".format(time['d']))
    logger.info("\twhitening:\t{0} seconds".format(time['w']))
    logger.info("\tsaving recording:\t{0} seconds".format(time['b']))
    logger.info("\tgetting waveforms:\t{0} seconds".format(time['e']))


    return score, clr_idx, spike_times


def process_batch(rec, get_score, BUFF, time, nnDetector, proj, nnTriage,
                  whiten_file):
    CONFIG = read_config()

    # detect spikes
    _b = datetime.datetime.now()
    if CONFIG.detctionMethod == 'nn':
        index = nnDetector.get_spikes(rec)
    else:

        index = threshold_detection(rec,
                                    CONFIG.neighChannels,
                                    CONFIG.spikeSize,
                                    CONFIG.stdFactor)

    # From Peter: When the recording is too long, I load them by
    # little chunk by chunk (chunk it time-wise). But I also add
    # some buffer. If the detected spike time is in the buffer,
    # i remove that because it will be detected in another chunk
    index = index[np.logical_and(index[:, 0] > BUFF,
                  index[:, 0] < (rec.shape[0] - BUFF))]

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
    chunk.astype(CONFIG.dtype).tofile(whiten_file)

    time['b'] += (datetime.datetime.now()-_b).total_seconds()

    _b = datetime.datetime.now()
    if CONFIG.detctionMethod == 'nn':
        score, clear_index, spike_times = get_waveforms(rec,
                                                        CONFIG.neighChannels,
                                                        index,
                                                        get_score,
                                                        proj,
                                                        CONFIG.spikeSize,
                                                        CONFIG.nFeat,
                                                        CONFIG.geom,
                                                        nnTriage,
                                                        CONFIG.nnThreshdoldCol)
        ss = 0
        nspikes = 0
    else:
        score, clear_index, spike_times = get_waveforms(rec,
                                                        CONFIG.neighChannels,
                                                        index,
                                                        0,
                                                        None,
                                                        CONFIG.spikeSize,
                                                        CONFIG.nFeat,
                                                        None,
                                                        None,
                                                        None)

        # TODO: ask peter, why is there a difference? getPcaSS is run
        # only when doing threshold detector, when doing nnet ss and
        # nspikes are 0
        ss, nspikes = getPcaSS(rec, spike_times, CONFIG.spikeSize,
                               CONFIG.BUFF)
    time['e'] += (datetime.datetime.now()-_b).total_seconds()

    return score, clear_index, spike_times, ss, nspikes, time
