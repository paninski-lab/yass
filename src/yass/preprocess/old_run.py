"""
Preprocess pipeline
"""
import datetime
import logging
import os.path

import numpy as np
import yaml

from .. import read_config
from ..batch import BatchPipeline, BatchProcessor, RecordingsReader
from ..batch import PipedTransformation as Transform

from .detect import threshold_detection
from .filter import localized_whitening_matrix, whitening_score, butterworth_single_channel
from .whitening import whitening_matrix, whitening
from .score import get_score_pca, get_pca_suff_stat, get_pca_projection
from .standarize import _standarize, standarize, sd
from ..neuralnetwork import NeuralNetDetector, NeuralNetTriage, nn_detection


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

    # logger = logging.getLogger(__name__)

    CONFIG = read_config()

    tmp = os.path.join(CONFIG.data.root_folder, 'tmp')

    if not os.path.exists(tmp):
        os.makedirs(tmp)

    # initialize processor for raw data
    path = os.path.join(CONFIG.data.root_folder, CONFIG.data.recordings)
    dtype = CONFIG.recordings.dtype

    pipeline = BatchPipeline(path, dtype, CONFIG.recordings.n_channels,
                             CONFIG.recordings.format,
                             CONFIG.resources.max_memory, tmp,
                             mode='single_channel_one_batch')

    if CONFIG.preprocess.filter:
        butterworth = Transform(butterworth_single_channel,
                                'filtered.bin', keep=True,
                                low_freq=CONFIG.filter.low_pass_freq,
                                high_factor=CONFIG.filter.high_factor,
                                order=CONFIG.filter.order,
                                sampling_freq=CONFIG.recordings.sampling_rate)

        pipeline.add([butterworth])

    standarize_op = Transform(_standarize, 'standarized.bin',
                              keep=True,
                              srate=CONFIG.recordings.sampling_rate)

    pipeline.add([standarize_op])

    # TODO: add whitening
    # what's the difference between whitening.localized_whitening_matrix
    # and whitening.whitening_matrix
    # whiten_file = open(os.path.join(CONFIG.data.root_folder,
    # 'tmp/whiten.bin'), 'wb')

    pipeline.run()

    path_to_standarized = os.path.join(tmp, 'standarized.bin')
    path_to_params = os.path.join(tmp, 'standarized.yaml')

    with open(path_to_params) as f:
        params = yaml.load(f)

    bp = BatchProcessor(path_to_standarized, params['dtype'],
                        params['n_channels'], params['data_format'],
                        CONFIG.resources.max_memory)

    gen = bp.multi_channel()

    standarized = RecordingsReader(path_to_standarized, mmap=False,
                                   output_shape='long')

    # run detector
    # nnet detector returns scores
    # threshold detector does not return scores, we need to compute them
    # in another function
    # can we split the logic? have one function for nn detection and another
    # one for nnet scoring

    # since the detector functions do not return one-to-one-data as in the
    # filter/standarize/whiten transformations we may need the user to define
    # a merge function to combine results from each batch

    # to make migration easier, let's start refactoring this on a single batch
    # run and then move to the batch processor implemetation
    # TODO: implement time logging inside batch processor
    # TODO: add single batch nndetector here
    # TODO: add single batch threshold detector here
    # TODO: add single batch threshold scoring here

    # if CONFIG.spikes.detection == 'nn':
    #     nnDetector = NeuralNetDetector(CONFIG.neural_network_detector.filename,
    #                                    CONFIG.neural_network_autoencoder.filename)
    #     nnTriage = NeuralNetTriage(CONFIG.neural_network_triage.filename)

    #     # TODO: there is some buffer logic here we need to remove...
    #     # detect spikes
    #     (spike_index_clear,
    #      spike_index_collision,
    #      score) = nn_detection(standarized, 10000, CONFIG.BUFF,
    #                            CONFIG.neighChannels, CONFIG.geom,
    #                            CONFIG.spikes.temporal_features, 3,
    #                            CONFIG.neural_network_detector.threshold_spike,
    #                            CONFIG.neural_network_triage.threshold_collision,
    #                            nnDetector,
    #                            nnTriage)
    spike_index = threshold_detection(standarized,
                                      CONFIG.neighChannels,
                                      CONFIG.spikeSize,
                                      CONFIG.stdFactor)

    # don't really understand the logic on this if statement...
    # if get_score ==0:
        # spike_index_clear = np.zeros((0,2), 'int32')
        # spike_index_collision = spike_index
    # else:
        # spike_index_clear = spike_index
        # spike_index_collision = np.zeros((0,2), 'int32')

    spike_index_clear = spike_index
    spike_index_collision = np.zeros((0, 2), 'int32')

    # score = None

    # get sufficient statistics for pca if we don't have projection matrix
    pca_suff_stat, spikes_per_channel = get_pca_suff_stat(standarized,
                                                          spike_index,
                                                          CONFIG.spikeSize)

    rot = get_pca_projection(pca_suff_stat, spikes_per_channel,
                                 CONFIG.spikes.temporal_features, CONFIG.neighChannels)
    score = get_score_pca(spike_index_clear, rot, CONFIG.neighChannels,
                              CONFIG.geom, CONFIG.batch_size,
                              CONFIG.BUFF, CONFIG.nBatches,
                              os.path.join(CONFIG.data.root_folder,'tmp/whiten.bin'),
                              CONFIG.scaleToSave)

    # TODO: remove spikes from buff area
    # TODO: spike time w.r.t. to the whole recording

    return (spike_index_clear, score, spike_index_collision, pca_suff_stat,
            spikes_per_channel)

    # initialize output variables
    get_score = 1
    spike_index_clear = None
    spike_index_collision = None
    score = None
    pca_suff_stat = None
    spikes_per_channel = None

    for i, batch in enumerate(gen):

        # TODO: ask peter, why would we only want to get score for a subset
        # of the data?
        # if i > CONFIG.nPortion:
            # get_score = 0

        # process batch
        # spike index is defined as a location in each minibatch
        (si_clr_batch, score_batch, si_col_batch,
         pss_batch, spc_batch) = process_batch(batch, CONFIG.BUFF)

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

    if CONFIG.spikes.detection != 'nn':
        _b = datetime.datetime.now()
        rot = get_pca_projection(pca_suff_stat, spikes_per_channel,
                                 CONFIG.spikes.temporal_features, CONFIG.neighChannels)
        score = get_score_pca(spike_index_clear, rot, CONFIG.neighChannels,
                              CONFIG.geom, CONFIG.batch_size,
                              CONFIG.BUFF, CONFIG.nBatches,
                              os.path.join(CONFIG.data.root_folder,'tmp/whiten.bin'),
                              CONFIG.scaleToSave)

    return score, spike_index_clear, spike_index_collision


def process_batch(rec, BUFF):
    CONFIG = read_config()

    # nn detection
    if CONFIG.spikes.detection == 'nn':
        nnDetector = NeuralNetDetector(CONFIG.neural_network_detector.filename,
                                       CONFIG.neural_network_autoencoder.filename)
        nnTriage = NeuralNetTriage(CONFIG.neural_network_triage.filename)

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

        if get_score ==0:
            spike_index_clear = np.zeros((0,2), 'int32')
            spike_index_collision = np.vstack((spike_index_collision,
                                       spike_index_clear))
            score = None
        
        else:
            # whiten signal
            _b = datetime.datetime.now()
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

        # whiten recording
        _b = datetime.datetime.now()

    # Remove spikes detectted in buffer area
    spike_index_clear = spike_index_clear[np.logical_and(
      spike_index_clear[:, 0] > BUFF, 
      spike_index_clear[:, 0] < (rec.shape[0] - BUFF))]
    spike_index_collision = spike_index_collision[np.logical_and(
      spike_index_collision[:, 0] > BUFF,
      spike_index_collision[:, 0] < (rec.shape[0] - BUFF))]

    _b = datetime.datetime.now()

    return (spike_index_clear, score, spike_index_collision,
        pca_suff_stat, spikes_per_channel)
