"""
Preprocess pipeline
"""
import logging
import os.path
from functools import reduce

import numpy as np

from .. import read_config
from ..batch import BatchPipeline, BatchProcessor
from ..batch import PipedTransformation as Transform

from .filter import butterworth
from .standarize import standarize
from . import whiten
from . import detect
from . import pca
from ..neuralnetwork import NeuralNetDetector, NeuralNetTriage, nn_detection


# TODO check legacy buffer logic
# TODO: fix indentation


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

    CONFIG = read_config()

    tmp = os.path.join(CONFIG.data.root_folder, 'tmp')

    if not os.path.exists(tmp):
        logger.info('Creating temporary folder: {}'.format(tmp))
        os.makedirs(tmp)
    else:
        logger.info('Temporary folder {} already exists, output will be '
                    'stored there'.format(tmp))

    path = os.path.join(CONFIG.data.root_folder, CONFIG.data.recordings)
    dtype = CONFIG.recordings.dtype

    # initialize pipeline object, one batch per channel
    pipeline = BatchPipeline(path, dtype, CONFIG.recordings.n_channels,
                             CONFIG.recordings.format,
                             CONFIG.resources.max_memory, tmp)

    # add filter transformation if necessary
    if CONFIG.preprocess.filter:
        filter_op = Transform(butterworth,
                              'filtered.bin',
                              mode='single_channel_one_batch',
                              keep=True,
                              low_freq=CONFIG.filter.low_pass_freq,
                              high_factor=CONFIG.filter.high_factor,
                              order=CONFIG.filter.order,
                              sampling_freq=CONFIG.recordings.sampling_rate)

        pipeline.add([filter_op])

    # standarize
    standarize_op = Transform(standarize, 'standarized.bin',
                              mode='single_channel_one_batch',
                              keep=True,
                              sampling_freq=CONFIG.recordings.sampling_rate)

    # whiten
    # TODO: add option to re-use Q
    whiten_op = Transform(whiten.apply, 'whitened.bin',
                          mode='multi_channel',
                          keep=True,
                          neighbors=CONFIG.neighChannels,
                          spike_size=CONFIG.spikeSize)

    pipeline.add([standarize_op, whiten_op])

    # run pipeline
    ((filtered_path, standarized_path, whitened_path),
     (filtered_params, standarized_params, whitened_params)) = pipeline.run()

    # detect spikes
    # TODO: support neural network, need to remove batch logic first
    bp = BatchProcessor(standarized_path, standarized_params['dtype'],
                        standarized_params['n_channels'],
                        standarized_params['data_format'],
                        CONFIG.resources.max_memory,
                        buffer_size=0)

    # apply threshold detector on standarized data
    spikes = bp.multi_channel_apply(detect.threshold,
                                    mode='memory',
                                    neighbors=CONFIG.neighChannels,
                                    spike_size=CONFIG.spikeSize,
                                    std_factor=CONFIG.stdFactor)
    spike_index_clear = np.vstack(spikes)

    # triage is not implemented on threshold detector, return empty array
    spike_index_collision = np.zeros((0, 2), 'int32')

    # compute per-batch sufficient statistics for PCA on standarized data
    stats = bp.multi_channel_apply(pca.suff_stat,
                                   mode='memory',
                                   spike_index=spike_index_clear,
                                   spike_size=CONFIG.spikeSize)

    suff_stats = reduce(lambda x, y: np.add(x, y), [e[0] for e in stats])

    spikes_per_channel = reduce(lambda x, y: np.add(x, y),
                                [e[1] for e in stats])

    # compute rotation matrix
    logger.info('Computing PCA projection matrix...')
    rotation = pca.project(suff_stats, spikes_per_channel,
                           CONFIG.spikes.temporal_features,
                           CONFIG.neighChannels)

    # TODO: make this parallel, we can split the spikes, generate batches
    # and score in parallel
    logger.info('Reducing spikes dimensionality with PCA matrix...')
    scores = pca.score(whitened_path, CONFIG.spikeSize, spike_index_clear,
                       rotation, CONFIG.neighChannels, CONFIG.geom)

    return scores, spike_index_clear, spike_index_collision
