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
from .filter import whitening_matrix, whitening, localized_whitening_matrix, whitening_score, butterworth
from .score import get_score_pca, get_pca_suff_stat, get_pca_projection
from .standarize import standarize, sd
from ..neuralnetwork import NeuralNetDetector, NeuralNetTriage, nn_detection

# remove this
Q = None
Q_score = None


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
    whiten_file = open(os.path.join(CONFIG.data.root_folder, 'tmp/whiten.bin'), 'wb')

    # initialize processor for raw data
    path = os.path.join(CONFIG.data.root_folder, CONFIG.data.recordings)
    dtype = CONFIG.recordings.dtype

    # initialize factory
    factory = BatchProcessorFactory(path_to_file=None,
                                    dtype=None,
                                    n_channels=CONFIG.recordings.n_channels,
                                    max_memory=CONFIG.resources.max_memory,
                                    buffer_size=None)

    if CONFIG.preprocess.filter == 1:

        _b = datetime.datetime.now()
        # make batch processor for raw data -> buterworth -> filtered
        bp = factory.make(path_to_file=path, dtype=dtype,
                          buffer_size=0)
        logger.info('Initialized butterworth batch processor: {}'
                    .format(bp))

        # run filtering
        path = os.path.join(CONFIG.data.root_folder,  'tmp/filtered.bin')
        dtype = bp.process_function(butterworth,
                                    path,
                                    CONFIG.filter.low_pass_freq,
                                    CONFIG.filter.high_factor,
                                    CONFIG.filter.order,
                                    CONFIG.recordings.sampling_rate)
        time['f'] += (datetime.datetime.now()-_b).total_seconds()

    # TODO: cache computations
    # make batch processor for filtered -> standarize -> standarized
    _b = datetime.datetime.now()
    bp = factory.make(path_to_file=path, dtype=dtype, buffer_size=0)

    # compute the standard deviation using the first batch only
    batch1 = next(bp)
    sd_ = sd(batch1, CONFIG.recordings.sampling_rate)

    # make another batch processor
    bp = factory.make(path_to_file=path, dtype=dtype, buffer_size=0)
    logger.info('Initialized standarization batch processor: {}'
                .format(bp))

    # run standarization
    path = os.path.join(CONFIG.data.root_folder,  'tmp/standarized.bin')
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
        if CONFIG.spikes.detection == 'nn':
            nnDetector = NeuralNetDetector(CONFIG.neural_network_detector.filename,
                                           CONFIG.neural_network_autoencoder.filename)
            nnTriage = NeuralNetTriage(CONFIG.neural_network_triage.filename)
            
        else:
            nnDetector = None
            nnTriage = None

        if i > CONFIG.nPortion:
            get_score = 0

        # process batch
        # spike index is defined as a location in each minibatch
        (si_clr_batch, score_batch, si_col_batch,
         pss_batch, spc_batch,
         time) = process_batch(batch, get_score, CONFIG.BUFF, time,
                               nnDetector=nnDetector,
                               nnTriage=nnTriage, whiten_file=whiten_file)

        # spike time w.r.t. to the whole recording
        si_clr_batch[:,0] = si_clr_batch[:,0] + i*CONFIG.batch_size - CONFIG.BUFF
        si_col_batch[:,0] = si_col_batch[:,0] + i*CONFIG.batch_size - CONFIG.BUFF
        
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
            if get_score == 1 and CONFIG.spikes.detection == 'nn':
                score = np.concatenate((score, score_batch), axis = 0)
            pca_suff_stat += pss_batch
            spikes_per_channel += spc_batch

    whiten_file.close()

    if CONFIG.spikes.detection != 'nn':
        _b = datetime.datetime.now()
        rot = get_pca_projection(pca_suff_stat, spikes_per_channel,
                                 CONFIG.spikes.temporal_features, CONFIG.neighChannels)
        score = get_score_pca(spike_index_clear, rot, CONFIG.neighChannels,
                              CONFIG.geom, CONFIG.batch_size,
                              CONFIG.BUFF, CONFIG.nBatches,
                              os.path.join(CONFIG.data.root_folder,'tmp/whiten.bin'),
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


def process_batch(rec, get_score, BUFF, time, nnDetector, nnTriage,
                  whiten_file):
    logger = logging.getLogger(__name__)
    CONFIG = read_config()

    global Q
    global Q_score
    
    # nn detection 
    if CONFIG.spikes.detection == 'nn':

        # detect spikes
        _b = datetime.datetime.now()
        (spike_index_clear, 
         spike_index_collision, 
         score) = nn_detection(rec, 10000, BUFF,
                               CONFIG.neighChannels,
                               CONFIG.geom,
                               CONFIG.spikes.temporal_features,
                               3,
                               CONFIG.neural_network_detector.threshold_spike,
                               CONFIG.neural_network_triage.threshold_collision,
                               nnDetector,
                               nnTriage
                              )

        # since we alread have scores, no need to calculated sufficient
        # statistics for pca
        pca_suff_stat = 0
        spikes_per_channel = 0

        time['d'] += (datetime.datetime.now()-_b).total_seconds()

        if get_score ==0:
            spike_index_clear = np.zeros((0,2), 'int32')
            spike_index_collision = np.vstack((spike_index_collision,
                                       spike_index_clear))
            score = None
        
        else:
            # whiten signal
            _b = datetime.datetime.now()
            # get withening matrix per batch or onece in total
            if CONFIG.preprocess.whiten_batchwise or Q is None:
                Q_score = localized_whitening_matrix(rec, 
                                               CONFIG.neighChannels, 
                                               CONFIG.geom, 
                                               CONFIG.spikeSize)
            score = whitening_score(score, spike_index_clear[:,1], Q_score)

            time['w'] += (datetime.datetime.now()-_b).total_seconds()


    # threshold detection
    elif CONFIG.spikes.detection == 'threshold':

        # detect spikes
        _b = datetime.datetime.now()
        spike_index = threshold_detection(rec,
                                          CONFIG.neighChannels,
                                          CONFIG.spikeSize,
                                          CONFIG.stdFactor)

        # every spikes are considered as clear spikes as no triage is done
        if get_score ==0:
            spike_index_clear = np.zeros((0,2), 'int32')
            spike_index_collision = spike_index
        else:
            spike_index_clear = spike_index
            spike_index_collision = np.zeros((0,2), 'int32')
        score = None

        # get sufficient statistics for pca if we don't have projection matrix
        pca_suff_stat, spikes_per_channel = get_pca_suff_stat(rec, spike_index,
                                                          CONFIG.spikeSize)

        time['d'] += (datetime.datetime.now()-_b).total_seconds()

        # whiten recording
        _b = datetime.datetime.now()
        if CONFIG.preprocess.whiten_batchwise or Q is None:
            Q = whitening_matrix(rec, CONFIG.neighChannels,
                                CONFIG.spikeSize)
        rec = whitening(rec, Q)

        time['w'] += (datetime.datetime.now()-_b).total_seconds()


    # Remove spikes detectted in buffer area
    spike_index_clear = spike_index_clear[np.logical_and(
      spike_index_clear[:, 0] > BUFF, 
      spike_index_clear[:, 0] < (rec.shape[0] - BUFF))]
    spike_index_collision = spike_index_collision[np.logical_and(
      spike_index_collision[:, 0] > BUFF,
      spike_index_collision[:, 0] < (rec.shape[0] - BUFF))]


    _b = datetime.datetime.now()

    # save whiten data
    chunk = rec*CONFIG.scaleToSave
    chunk.reshape(chunk.shape[0]*chunk.shape[1])
    chunk.astype('int16').tofile(whiten_file)

    time['b'] += (datetime.datetime.now()-_b).total_seconds()

    return (spike_index_clear, score, spike_index_collision,
        pca_suff_stat, spikes_per_channel, time)
