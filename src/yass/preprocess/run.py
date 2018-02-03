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
from .standarize import standarize, standard_deviation
from . import whiten
from . import detect
from . import dimensionality_reduction as dim_red
from .. import neuralnetwork
import scipy.spatial as ss


def run(output_directory='tmp/'):
    """Execute preprocessing pipeline

    Parameters
    ----------
    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/

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
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/:

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

    OUTPUT_DTYPE = CONFIG.preprocess.dtype

    logger.info('Output dtype for transformed data will be {}'
                .format(OUTPUT_DTYPE))

    TMP = os.path.join(CONFIG.data.root_folder, output_directory)

    if not os.path.exists(TMP):
        logger.info('Creating temporary folder: {}'.format(TMP))
        os.makedirs(TMP)
    else:
        logger.info('Temporary folder {} already exists, output will be '
                    'stored there'.format(TMP))

    path = os.path.join(CONFIG.data.root_folder, CONFIG.data.recordings)
    dtype = CONFIG.recordings.dtype

    # initialize pipeline object, one batch per channel
    pipeline = BatchPipeline(path, dtype, CONFIG.recordings.n_channels,
                             CONFIG.recordings.format,
                             CONFIG.resources.max_memory, TMP)

    # add filter transformation if necessary
    if CONFIG.preprocess.filter:
        filter_op = Transform(
            butterworth,
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

    (filtered_path,), (filtered_params,) = pipeline.run()

    # standarize
    bp = BatchProcessor(
        filtered_path, filtered_params['dtype'], filtered_params['n_channels'],
        filtered_params['data_format'], CONFIG.resources.max_memory)
    batches = bp.multi_channel()
    first_batch, _, _ = next(batches)
    sd = standard_deviation(first_batch, CONFIG.recordings.sampling_rate)

    (standarized_path, standarized_params) = bp.multi_channel_apply(
        standarize,
        mode='disk',
        output_path=os.path.join(TMP, 'standarized.bin'),
        if_file_exists='skip',
        cast_dtype=OUTPUT_DTYPE,
        sd=sd)

    standarized = RecordingsReader(standarized_path)
    n_observations = standarized.observations

    if CONFIG.spikes.detection == 'threshold':
        return _threshold_detection(standarized_path, standarized_params,
                                    n_observations, output_directory)
    elif CONFIG.spikes.detection == 'nn':
        return _neural_network_detection(standarized_path, standarized_params,
                                         n_observations, output_directory)


def _threshold_detection(standarized_path, standarized_params, n_observations,
                         output_directory):
    """Run threshold detector and dimensionality reduction using PCA
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    OUTPUT_DTYPE = CONFIG.preprocess.dtype
    TMP_FOLDER = os.path.join(CONFIG.data.root_folder, output_directory)

    ###############
    # Whiten data #
    ###############

    # compute Q for whitening
    logger.info('Computing whitening matrix...')
    bp = BatchProcessor(standarized_path, standarized_params['dtype'],
                        standarized_params['n_channels'],
                        standarized_params['data_format'],
                        CONFIG.resources.max_memory)
    batches = bp.multi_channel()
    first_batch, _, _ = next(batches)
    Q = whiten.matrix(first_batch, CONFIG.neighChannels, CONFIG.spikeSize)

    path_to_whitening_matrix = os.path.join(TMP_FOLDER, 'whitening.npy')
    np.save(path_to_whitening_matrix, Q)
    logger.info('Saved whitening matrix in {}'
                .format(path_to_whitening_matrix))

    # apply whitening to every batch
    (whitened_path, whitened_params) = bp.multi_channel_apply(
        np.matmul,
        mode='disk',
        output_path=os.path.join(TMP_FOLDER, 'whitened.bin'),
        if_file_exists='skip',
        cast_dtype=OUTPUT_DTYPE,
        b=Q)

    ###################
    # Spike detection #
    ###################

    path_to_spike_index_clear = os.path.join(TMP_FOLDER,
                                             'spike_index_clear.npy')

    bp = BatchProcessor(
        standarized_path,
        standarized_params['dtype'],
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
                    ' detector...'.format(path_to_spike_index_clear))

        # apply threshold detector on standarized data
        spikes = bp.multi_channel_apply(
            detect.threshold,
            mode='memory',
            cleanup_function=detect.fix_indexes,
            neighbors=CONFIG.neighChannels,
            spike_size=CONFIG.spikeSize,
            std_factor=CONFIG.stdFactor)
        spike_index_clear = np.vstack(spikes)

        logger.info('Removing clear indexes outside the allowed range to '
                    'draw a complete waveform...')
        spike_index_clear, _ = (detect.remove_incomplete_waveforms(
            spike_index_clear, CONFIG.spikeSize + CONFIG.templatesMaxShift,
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

    # load and dump waveforms from clear spikes
    path_to_waveforms_clear = os.path.join(TMP_FOLDER, 'waveforms_clear.npy')

    if os.path.exists(path_to_waveforms_clear):
        logger.info('Found clear waveforms in {}, loading them...'
                    .format(path_to_waveforms_clear))
        waveforms_clear = np.load(path_to_waveforms_clear)
    else:
        logger.info('Did not find clear waveforms in {}, reading them from {}'
                    .format(path_to_waveforms_clear, standarized_path))
        explorer = RecordingExplorer(
            standarized_path, spike_size=CONFIG.spikeSize)
        waveforms_clear = explorer.read_waveforms(spike_index_clear[:, 0])
        np.save(path_to_waveforms_clear, waveforms_clear)
        logger.info('Saved waveform from clear spikes in: {}'
                    .format(path_to_waveforms_clear))

    #########################
    # PCA - rotation matrix #
    #########################

    # compute per-batch sufficient statistics for PCA on standarized data
    logger.info('Computing PCA sufficient statistics...')
    stats = bp.multi_channel_apply(
        dim_red.suff_stat,
        mode='memory',
        spike_index=spike_index_clear,
        spike_size=CONFIG.spikeSize)

    suff_stats = reduce(lambda x, y: np.add(x, y), [e[0] for e in stats])

    spikes_per_channel = reduce(lambda x, y: np.add(x, y),
                                [e[1] for e in stats])

    # compute rotation matrix
    logger.info('Computing PCA projection matrix...')
    rotation = dim_red.project(suff_stats, spikes_per_channel,
                               CONFIG.spikes.temporal_features,
                               CONFIG.neighChannels)
    path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')
    np.save(path_to_rotation, rotation)
    logger.info('Saved rotation matrix in {}...'.format(path_to_rotation))

    ###########################################
    # PCA - waveform dimensionality reduction #
    ###########################################

    logger.info('Reducing spikes dimensionality with PCA matrix...')
    scores = dim_red.score(waveforms_clear, rotation, spike_index_clear[:, 1],
                           CONFIG.neighChannels, CONFIG.geom)

    # save scores
    path_to_score = os.path.join(TMP_FOLDER, 'score_clear.npy')
    np.save(path_to_score, scores)
    logger.info('Saved spike scores in {}...'.format(path_to_score))

    return scores, spike_index_clear, spike_index_collision


def _neural_network_detection(standarized_path, standarized_params,
                              n_observations, output_directory):
    """Run neural network detection and autoencoder dimensionality reduction
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    OUTPUT_DTYPE = CONFIG.preprocess.dtype
    TMP_FOLDER = os.path.join(CONFIG.data.root_folder, output_directory)

    # detect spikes
    bp = BatchProcessor(
        standarized_path,
        standarized_params['dtype'],
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

    if all([
        os.path.exists(path_to_score),
        os.path.exists(path_to_spike_index_clear),
        os.path.exists(path_to_spike_index_collision)
    ]):
        logger.info('Loading "{}", "{}" and "{}"'.format(
            path_to_score, path_to_spike_index_clear,
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
        res = mc(
            neuralnetwork.nn_detection,
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
        clear, idx = detect.remove_incomplete_waveforms(
            clear, CONFIG.spikeSize + CONFIG.templatesMaxShift,
            n_observations)
        np.save(path_to_spike_index_clear, clear)
        logger.info('Saved spike index clear in {}...'
                    .format(path_to_spike_index_clear))

        # save collided spikes
        collision = np.concatenate([element[2] for element in res], axis=0)
        logger.info('Removing collision indexes outside the allowed range to '
                    'draw a complete waveform...')
        collision, _ = detect.remove_incomplete_waveforms(
            collision, CONFIG.spikeSize + CONFIG.templatesMaxShift,
            n_observations)
        np.save(path_to_spike_index_collision, collision)
        logger.info('Saved spike index collision in {}...'
                    .format(path_to_spike_index_collision))

        if CONFIG.clustering.clustering_method == '2+3':
            #######################
            # Waveform extraction #
            #######################

            # TODO: what should the behaviour be for spike indexes that are
            # when starting/ending the recordings and it is not possible to
            # draw a complete waveform?
            logger.info('Computing whitening matrix...')
            bp = BatchProcessor(standarized_path, standarized_params['dtype'],
                                standarized_params['n_channels'],
                                standarized_params['data_format'],
                                CONFIG.resources.max_memory)
            batches = bp.multi_channel()
            first_batch, _, _ = next(batches)
            Q = whiten.matrix(first_batch, CONFIG.neighChannels,
                              CONFIG.spikeSize)

            path_to_whitening_matrix = os.path.join(TMP_FOLDER,
                                                    'whitening.npy')
            np.save(path_to_whitening_matrix, Q)
            logger.info('Saved whitening matrix in {}'
                        .format(path_to_whitening_matrix))

            # apply whitening to every batch
            (whitened_path, whitened_params) = bp.multi_channel_apply(
                np.matmul,
                mode='disk',
                output_path=os.path.join(TMP_FOLDER, 'whitened.bin'),
                if_file_exists='skip',
                cast_dtype=OUTPUT_DTYPE,
                b=Q)

            main_channel = clear[:, 1]

            # load and dump waveforms from clear spikes

            path_to_waveforms_clear = os.path.join(TMP_FOLDER,
                                                   'waveforms_clear.npy')

            if os.path.exists(path_to_waveforms_clear):
                logger.info('Found clear waveforms in {}, loading them...'
                            .format(path_to_waveforms_clear))
                waveforms_clear = np.load(path_to_waveforms_clear)
            else:
                logger.info(
                    'Did not find clear waveforms in {}, reading them from {}'
                    .format(path_to_waveforms_clear, whitened_path))
                explorer = RecordingExplorer(
                    whitened_path, spike_size=CONFIG.spikeSize)
                waveforms_clear = explorer.read_waveforms(clear[:, 0], 'all')
                np.save(path_to_waveforms_clear, waveforms_clear)
                logger.info('Saved waveform from clear spikes in: {}'
                            .format(path_to_waveforms_clear))

            main_channel = clear[:, 1]

            # save rotation
            detector_filename = CONFIG.neural_network_detector.filename
            autoencoder_filename = CONFIG.neural_network_autoencoder.filename
            rotation = neuralnetwork.load_rotation(detector_filename,
                                                   autoencoder_filename)
            path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')
            logger.info("rotation_matrix_shape = {}".format(rotation.shape))
            np.save(path_to_rotation, rotation)
            logger.info(
                'Saved rotation matrix in {}...'.format(path_to_rotation))

            logger.info('Denoising...')
            path_to_denoised_waveforms = os.path.join(TMP_FOLDER,
                                                      'denoised_waveforms.npy')
            if os.path.exists(path_to_denoised_waveforms):
                logger.info('Found denoised waveforms in {}, loading them...'
                            .format(path_to_denoised_waveforms))
                denoised_waveforms = np.load(path_to_denoised_waveforms)
            else:
                logger.info(
                    'Did not find denoised waveforms in {}, evaluating them'
                    'from {}'.format(path_to_denoised_waveforms,
                                     path_to_waveforms_clear))
                waveforms_clear = np.load(path_to_waveforms_clear)
                denoised_waveforms = dim_red.denoise(waveforms_clear, rotation,
                                                     CONFIG)
                logger.info('Saving denoised waveforms to {}'.format(
                    path_to_denoised_waveforms))
                np.save(path_to_denoised_waveforms, denoised_waveforms)

            isolated_index, x, y = get_isolated_spikes_and_locations(
                denoised_waveforms, main_channel, CONFIG)
            x = (x - np.mean(x)) / np.std(x)
            y = (y - np.mean(y)) / np.std(y)
            corrupted_index = np.logical_not(
                np.in1d(np.arange(clear.shape[0]), isolated_index))
            collision = np.concatenate(
                [collision, clear[corrupted_index]], axis=0)
            clear = clear[isolated_index]
            waveforms_clear = waveforms_clear[isolated_index]

            #################################################
            # Dimensionality reduction (Isolated Waveforms) #
            #################################################

            scores = dim_red.main_channel_scores(waveforms_clear, rotation,
                                                 clear, CONFIG)
            scores = (scores - np.mean(scores, axis=0)) / np.std(scores)
            scores = np.concatenate(
                [
                    x[:, np.newaxis, np.newaxis], y[:, np.newaxis, np.newaxis],
                    scores[:, :, np.newaxis]
                ],
                axis=1)

        else:

            # save scores
            scores = np.concatenate([element[0] for element in res], axis=0)

            logger.info(
                'Removing scores for indexes outside the allowed range to '
                'draw a complete waveform...')
            scores = scores[idx]

            # compute Q for whitening
            logger.info('Computing whitening matrix...')
            bp = BatchProcessor(standarized_path, standarized_params['dtype'],
                                standarized_params['n_channels'],
                                standarized_params['data_format'],
                                CONFIG.resources.max_memory)
            batches = bp.multi_channel()
            first_batch, _, _ = next(batches)
            Q = whiten.matrix_localized(first_batch, CONFIG.neighChannels,
                                        CONFIG.geom, CONFIG.spikeSize)

            path_to_whitening_matrix = os.path.join(TMP_FOLDER,
                                                    'whitening.npy')
            np.save(path_to_whitening_matrix, Q)
            logger.info('Saved whitening matrix in {}'
                        .format(path_to_whitening_matrix))

            scores = whiten.score(scores, clear[:, 1], Q)

            np.save(path_to_score, scores)
            logger.info('Saved spike scores in {}...'.format(path_to_score))

            # save rotation
            detector_filename = CONFIG.neural_network_detector.filename
            autoencoder_filename = CONFIG.neural_network_autoencoder.filename
            rotation = neuralnetwork.load_rotation(detector_filename,
                                                   autoencoder_filename)
            path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')
            np.save(path_to_rotation, rotation)
            logger.info(
                'Saved rotation matrix in {}...'.format(path_to_rotation))

        np.save(path_to_score, scores)
        logger.info('Saved spike scores in {}...'.format(path_to_score))

    return scores, clear, collision


def get_isolated_spikes_and_locations(denoised_waveforms, main_channel,
                                      CONFIG):
    power_all_chan_denoised = np.linalg.norm(denoised_waveforms, axis=1)
    th = CONFIG.isolation_threshold
    isolated_index = []
    dist = ss.distance_matrix(CONFIG.geom, CONFIG.geom)
    for i in range(denoised_waveforms.shape[0] - 1):
        if dist[main_channel[i], main_channel[i + 1]] > th:
            if i == 0:
                isolated_index += [i]
            elif dist[main_channel[i], main_channel[i - 1]] > th:
                isolated_index += [i]
    isolated_index = np.asarray(isolated_index, dtype=np.int32)

    mask = np.ones([isolated_index.shape[0], CONFIG.recordings.n_channels])
    for i, j in enumerate(isolated_index):
        nn = np.where(CONFIG.neighChannels[main_channel[j]])[0]
        for k in range(CONFIG.recordings.n_channels):
            if mask[i, k] and k not in nn:
                mask[i, k] = 0

    x = np.sum(
        mask * power_all_chan_denoised[isolated_index] * CONFIG.geom[:, 0],
        axis=1) / np.sum(
        mask * power_all_chan_denoised[isolated_index], axis=1)
    y = np.sum(
        mask * power_all_chan_denoised[isolated_index] * CONFIG.geom[:, 1],
        axis=1) / np.sum(
        mask * power_all_chan_denoised[isolated_index], axis=1)

    return isolated_index, x, y
