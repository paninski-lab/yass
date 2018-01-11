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
from ..explore import RecordingExplorer

from .filter import butterworth
from .standarize import standarize
from . import whiten
from . import detect
from . import pca
from .. import neuralnetwork


def run():
    """Execute preprocessing pipeline

    Returns
    -------
    scores: numpy.ndarray (n_spikes, n_features, n_channels)
        List of size n_channels, each list contains a (clear spikes x
        number of features x number of channels) multidimensional array
        score for every clear spike

    clear_index: numpy.ndarray
        List of size n_channels, each list contains the indexes in
        spike_times (first column) where the spike was clear

    spike_times: numpy.ndarray
        List with n_channels elements, each element contains spike times
        in the first column and [SECOND COLUMN?]

    Examples
    --------

    .. literalinclude:: ../examples/preprocess.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    OUTPUT_DTYPE = 'float16'

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
                              if_file_exists='skip',
                              cast_dtype=OUTPUT_DTYPE,
                              low_freq=CONFIG.filter.low_pass_freq,
                              high_factor=CONFIG.filter.high_factor,
                              order=CONFIG.filter.order,
                              sampling_freq=CONFIG.recordings.sampling_rate)

        pipeline.add([filter_op])

    # standarize
    standarize_op = Transform(standarize, 'standarized.bin',
                              mode='single_channel_one_batch',
                              keep=True,
                              if_file_exists='skip',
                              cast_dtype=OUTPUT_DTYPE,
                              sampling_freq=CONFIG.recordings.sampling_rate)

    # whiten
    # TODO: add option to re-use Q
    whiten_op = Transform(whiten.apply, 'whitened.bin',
                          mode='multi_channel',
                          keep=True,
                          if_file_exists='skip',
                          cast_dtype=OUTPUT_DTYPE,
                          neighbors=CONFIG.neighChannels,
                          spike_size=CONFIG.spikeSize)

    pipeline.add([standarize_op, whiten_op])

    # run pipeline
    ((filtered_path, standarized_path, whitened_path),
     (filtered_params, standarized_params, whitened_params)) = pipeline.run()

    if CONFIG.spikes.detection == 'threshold':
        return _threshold_detection(standarized_path, standarized_params,
                                    whitened_path)
    elif CONFIG.spikes.detection == 'nn':
        return _neural_network_detection(standarized_path, standarized_params)


def _threshold_detection(standarized_path, standarized_params, whitened_path):
    """Run threshold detector and dimensionality reduction using PCA
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    TMP_FOLDER = os.path.join(CONFIG.data.root_folder, 'tmp/')

    path_to_spike_index_clear = os.path.join(CONFIG.data.root_folder, 'tmp',
                                             'spike_index_clear.npy')

    bp = BatchProcessor(standarized_path, standarized_params['dtype'],
                        standarized_params['n_channels'],
                        standarized_params['data_format'],
                        CONFIG.resources.max_memory,
                        buffer_size=0)

    # check if spike_index_clear exists...
    if os.path.exists(path_to_spike_index_clear):
        # if it exists, load it...
        logger.info('Found file in {}, loading it...'
                    .format(path_to_spike_index_clear))
        spike_index_clear = np.load(path_to_spike_index_clear)
    else:
        # if it doesn't, detect spikes...
        logger.info('Did not find file in {}, finding spikes using threshold'
                    ' detector...'
                    .format(path_to_spike_index_clear))

        # apply threshold detector on standarized data
        spikes = bp.multi_channel_apply(detect.threshold,
                                        mode='memory',
                                        cleanup_function=detect.fix_indexes,
                                        neighbors=CONFIG.neighChannels,
                                        spike_size=CONFIG.spikeSize,
                                        std_factor=CONFIG.stdFactor)
        spike_index_clear = np.vstack(spikes)

        logger.info('Saving spikes in {}...'.format(path_to_spike_index_clear))
        np.save(path_to_spike_index_clear, spike_index_clear)

    # triage is not implemented on threshold detector, return empty array
    spike_index_collision = np.zeros((0, 2), 'int32')

    # load and dump waveforms from clear spikes
    explorer = RecordingExplorer(whitened_path, spike_size=CONFIG.spikeSize)
    waveforms_clear = explorer.read_waveforms(spike_index_clear[:, 0])
    path_to_waveforms_clear = os.path.join(TMP_FOLDER, 'waveforms_clear.npy')
    np.save(path_to_waveforms_clear, waveforms_clear)
    logger.info('Saved waveform from clear spikes in: {}'
                .format(path_to_waveforms_clear))

    # compute per-batch sufficient statistics for PCA on standarized data
    logger.info('Computing PCA sufficient statistics...')
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
    path_to_rotation = os.path.join(CONFIG.data.root_folder, 'tmp',
                                    'rotation.npy')
    np.save(path_to_rotation, rotation)
    logger.info('Saved rotation matrix in {}...'.format(path_to_rotation))

    logger.info('Reducing spikes dimensionality with PCA matrix...')
    scores = pca.score(waveforms_clear, spike_index_clear, rotation,
                       CONFIG.neighChannels, CONFIG.geom)

    # save scores
    path_to_score = os.path.join(CONFIG.data.root_folder, 'tmp', 'score.npy')
    np.save(path_to_score, scores)
    logger.info('Saved spike scores in {}...'.format(path_to_score))

    return scores, spike_index_clear, spike_index_collision


def _neural_network_detection(standarized_path, standarized_params):
    """Run neural network detection and autoencoder dimensionality reduction
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    # detect spikes
    bp = BatchProcessor(standarized_path, standarized_params['dtype'],
                        standarized_params['n_channels'],
                        standarized_params['data_format'],
                        CONFIG.resources.max_memory,
                        buffer_size=0)

    # apply threshold detector on standarized data
    mc = bp.multi_channel_apply
    res = mc(neuralnetwork.nn_detection,
             mode='memory',
             cleanup_function=neuralnetwork.fix_indexes,
             neighbors=CONFIG.neighChannels,
             geom=CONFIG.geom,
             temporal_features=CONFIG.spikes.temporal_features,
             # FIXME: what is this?
             temporal_window=3,
             th_detect=CONFIG.neural_network_detector.threshold_spike,
             th_triage=CONFIG.neural_network_triage.threshold_collision,
             detector_filename=CONFIG.neural_network_detector.filename,
             autoencoder_filename=CONFIG.neural_network_autoencoder.filename,
             triage_filename=CONFIG.neural_network_triage.filename)

    scores = np.concatenate([element[0] for element in res], axis=0)

    # save scores
    path_to_score = os.path.join(CONFIG.data.root_folder, 'tmp', 'score.npy')
    np.save(path_to_score, scores)
    logger.info('Saved spike scores in {}...'.format(path_to_score))

    # save rotation
    detector_filename = CONFIG.neural_network_detector.filename
    autoencoder_filename = CONFIG.neural_network_autoencoder.filename
    rotation = neuralnetwork.load_rotation(detector_filename,
                                           autoencoder_filename)
    path_to_rotation = os.path.join(CONFIG.data.root_folder, 'tmp',
                                    'rotation.npy')
    np.save(path_to_rotation, rotation)
    logger.info('Saved rotation matrix in {}...'.format(path_to_rotation))

    clear = np.concatenate([element[1] for element in res], axis=0)
    collision = np.concatenate([element[2] for element in res], axis=0)

    return scores, clear, collision
