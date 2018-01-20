"""
Preprocess pipeline
"""
import logging
import os.path
from functools import reduce

import numpy as np

from .. import read_config
from ..batch import BatchPipeline, BatchProcessor, RecordingsReader
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
    clear_scores: numpy.ndarray (n_spikes, n_features, n_channels)
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels

    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    spike_index_collision: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for collided spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    Notes
    -----
    Running the preprocessor will generate some files in
    CONFIG.data.root_folder/tmp/, the files are the following:

    * ``config.yaml`` - Copy of the configuration file
    * ``metadata.yaml`` - Experiment metadata
    * ``filtered.bin`` - Filtered recordings
    * ``filtered.yaml`` - Filtered recordings metadata
    * ``standarized.bin`` - Standarized recordings
    * ``standarized.yaml`` - Standarized recordings metadata
    * ``whitened.bin`` - Whitened recordings
    * ``whitened.yaml`` - Whitened recordings metadata
    * ``rotation.npy`` - Rotation matrix for dimensionality reduction
    * ``spike_index_clear.npy`` - Same as spike_index_clear returned
    * ``spike_index_collision.npy`` - Same as spike_index_collision returned
    * ``score_clear.npy`` - Scores for clear spikes
    * ``waveforms_clear.npy`` - Waveforms for clear spikes

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

    standarized = RecordingsReader(standarized_path)
    n_observations = standarized.observations
    del standarized

    if CONFIG.spikes.detection == 'threshold':
        return _threshold_detection(standarized_path, standarized_params,
                                    whitened_path, n_observations)
    elif CONFIG.spikes.detection == 'nn':
        return _neural_network_detection(standarized_path, standarized_params,
                                         n_observations)


def _threshold_detection(standarized_path, standarized_params, whitened_path,
                         n_observations):
    """Run threshold detector and dimensionality reduction using PCA
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    TMP_FOLDER = os.path.join(CONFIG.data.root_folder, 'tmp/')

    ###################
    # Spike detection #
    ###################

    path_to_spike_index_clear = os.path.join(TMP_FOLDER,
                                             'spike_index_clear.npy')

    bp = BatchProcessor(standarized_path, standarized_params['dtype'],
                        standarized_params['n_channels'],
                        standarized_params['data_format'],
                        CONFIG.resources.max_memory,
                        buffer_size=0)

    # clear spikes
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

        logger.info('Removing clear indexes outside the allowed range to '
                    'draw a complete waveform...')
        spike_index_clear, _ = (detect
                                .remove_incomplete_waveforms(spike_index_clear,
                                                             CONFIG.spikeSize,
                                                             n_observations))

        logger.info('Saving spikes in {}...'.format(path_to_spike_index_clear))
        np.save(path_to_spike_index_clear, spike_index_clear)

    path_to_spike_index_collision = os.path.join(TMP_FOLDER,
                                                 'spike_index_collision.npy')

    # collided spikes
    if os.path.exists(path_to_spike_index_collision):
        # if it exists, load it...
        logger.info('Found collided spikes in {}, loading them...'
                    .format(path_to_spike_index_collision))
        spike_index_collision = np.load(path_to_spike_index_collision)

        if spike_index_collision.shape[0] != 0:
            raise ValueError('Found non-empty collision spike index in {}, '
                             'but threshold detector is selected, collision '
                             'detection is not implemented for threshold '
                             'detector so array must have dimensios (0, 2) '
                             'but had ({}, {})'
                             .format(path_to_spike_index_collision,
                                     *spike_index_collision.shape))
    else:
        # triage is not implemented on threshold detector, return empty array
        logger.info('Creating empty array for'
                    ' collided spikes (collision detection is not implemented'
                    ' with threshold detector. Saving them in {}'
                    .format(path_to_spike_index_collision))
        spike_index_collision = np.zeros((0, 2), 'int32')
        np.save(path_to_spike_index_collision, spike_index_collision)

    #######################
    # Waveform extraction #
    #######################

    # TODO: what should the behaviour be for spike indexes that are when
    # starting/ending the recordings and it is not possible ti draw a complete
    # waveform?

    # load and dump waveforms from clear spikes
    path_to_waveforms_clear = os.path.join(TMP_FOLDER, 'waveforms_clear.npy')

    if os.path.exists(path_to_waveforms_clear):
        logger.info('Found clear waveforms in {}, loading them...'
                    .format(path_to_waveforms_clear))
        waveforms_clear = np.load(path_to_waveforms_clear)
    else:
        logger.info('Did not find clear waveforms in {}, reading them from {}'
                    .format(path_to_waveforms_clear, whitened_path))
        explorer = RecordingExplorer(whitened_path,
                                     spike_size=CONFIG.spikeSize)
        waveforms_clear = explorer.read_waveforms(spike_index_clear[:, 0])
        np.save(path_to_waveforms_clear, waveforms_clear)
        logger.info('Saved waveform from clear spikes in: {}'
                    .format(path_to_waveforms_clear))

    #########################
    # PCA - rotation matrix #
    #########################

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
    path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')
    np.save(path_to_rotation, rotation)
    logger.info('Saved rotation matrix in {}...'.format(path_to_rotation))

    ###########################################
    # PCA - waveform dimensionality reduction #
    ###########################################

    logger.info('Reducing spikes dimensionality with PCA matrix...')
    scores = pca.score(waveforms_clear, spike_index_clear, rotation,
                       CONFIG.neighChannels, CONFIG.geom)

    # save scores
    path_to_score = os.path.join(TMP_FOLDER, 'score_clear.npy')
    np.save(path_to_score, scores)
    logger.info('Saved spike scores in {}...'.format(path_to_score))

    return scores, spike_index_clear, spike_index_collision


def _neural_network_detection(standarized_path, standarized_params,
                              n_observations):
    """Run neural network detection and autoencoder dimensionality reduction
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    TMP_FOLDER = os.path.join(CONFIG.data.root_folder, 'tmp/')

    # detect spikes
    bp = BatchProcessor(standarized_path, standarized_params['dtype'],
                        standarized_params['n_channels'],
                        standarized_params['data_format'],
                        CONFIG.resources.max_memory,
                        buffer_size=0)

    # check if all scores, clear and collision spikes exist..
    path_to_score = os.path.join(TMP_FOLDER, 'score_clear.npy')
    path_to_spike_index_clear = os.path.join(TMP_FOLDER,
                                             'spike_index_clear.npy')
    path_to_spike_index_collision = os.path.join(TMP_FOLDER,
                                                 'spike_index_collision.npy')

    if all([os.path.exists(path_to_score),
            os.path.exists(path_to_spike_index_clear),
            os.path.exists(path_to_spike_index_collision)]):
        logger.info('Loading "{}", "{}" and "{}"'
                    .format(path_to_score,
                            path_to_spike_index_clear,
                            path_to_spike_index_collision))

        scores = np.load(path_to_score)
        clear = np.load(path_to_spike_index_clear)
        collision = np.load(path_to_spike_index_collision)

    else:
        logger.info('One or more of "{}", "{}" or "{}" files were missing, '
                    'computing...'.format(path_to_score,
                                          path_to_spike_index_clear,
                                          path_to_spike_index_collision))

        # apply threshold detector on standarized data
        autoencoder_filename = CONFIG.neural_network_autoencoder.filename
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
                 autoencoder_filename=autoencoder_filename,
                 triage_filename=CONFIG.neural_network_triage.filename)

        # save clear spikes
        clear = np.concatenate([element[1] for element in res], axis=0)
        logger.info('Removing clear indexes outside the allowed range to '
                    'draw a complete waveform...')
        clear, idx = detect.remove_incomplete_waveforms(clear,
                                                        CONFIG.spikeSize,
                                                        n_observations)
        np.save(path_to_spike_index_clear, clear)
        logger.info('Saved spike index clear in {}...'
                    .format(path_to_spike_index_clear))

        # save collided spikes
        collision = np.concatenate([element[2] for element in res], axis=0)
        logger.info('Removing collision indexes outside the allowed range to '
                    'draw a complete waveform...')
        collision, _ = detect.remove_incomplete_waveforms(collision,
                                                          CONFIG.spikeSize,
                                                          n_observations)
        np.save(path_to_spike_index_collision, collision)
        logger.info('Saved spike index collision in {}...'
                    .format(path_to_spike_index_collision))

        # save scores
        scores = np.concatenate([element[0] for element in res], axis=0)
        logger.info('Removing scores for indexes outside the allowed range to '
                    'draw a complete waveform...')
        scores = scores[idx]
        np.save(path_to_score, scores)
        logger.info('Saved spike scores in {}...'.format(path_to_score))

        # save rotation
        detector_filename = CONFIG.neural_network_detector.filename
        autoencoder_filename = CONFIG.neural_network_autoencoder.filename
        rotation = neuralnetwork.load_rotation(detector_filename,
                                               autoencoder_filename)
        path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')
        np.save(path_to_rotation, rotation)
        logger.info('Saved rotation matrix in {}...'.format(path_to_rotation))

    return scores, clear, collision
