"""
Preprocess pipeline
"""
import logging
import os.path
from functools import reduce

import numpy as np

from yass import read_config
from yass.batch import BatchPipeline, BatchProcessor, RecordingsReader
from yass.batch import PipedTransformation as Transform
from yass.geometry import make_channel_index

from yass.preprocess.filter import butterworth
from yass.preprocess.standarize import standarize, standard_deviation
from yass.preprocess import whiten
from yass.threshold import detect
from yass.threshold import dimensionality_reduction as dim_red
from yass import neuralnetwork


def run(output_directory='tmp/'):
    """Execute preprocessing pipeline

    Parameters
    ----------
    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/

    Returns
    -------
    scores: numpy.ndarray (n_spikes, n_features, n_channels)
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels

    clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    collision: numpy.ndarray (n_collided_spikes, 2)
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

    # Whiten
    bp = BatchProcessor(
        standarized_path,
        standarized_params['dtype'],
        standarized_params['n_channels'],
        standarized_params['data_format'],
        CONFIG.resources.max_memory)

    # determine neighboring channel info
    channel_index = make_channel_index(CONFIG.neighChannels,
                                       CONFIG.geom)

    # compute whiten_filter (Q) for whitening
    logger.info('Computing whitening matrix...')
    batches = bp.multi_channel()
    first_batch, _, _ = next(batches)
    whiten_filter = whiten.matrix(first_batch, channel_index,
                                  CONFIG.spikeSize)

    path_to_whitening_matrix = os.path.join(TMP, 'whitening.npy')
    np.save(path_to_whitening_matrix, whiten_filter)
    logger.info('Saved whitening matrix in {}'
                .format(path_to_whitening_matrix))

    # run detection
    if CONFIG.spikes.detection == 'threshold':
        return _threshold_detection(standarized_path,
                                    standarized_params,
                                    channel_index,
                                    whiten_filter,
                                    output_directory)
    elif CONFIG.spikes.detection == 'nn':
        return _neural_network_detection(standarized_path,
                                         standarized_params,
                                         channel_index,
                                         whiten_filter,
                                         output_directory)


def _threshold_detection(standarized_path, standarized_params, channel_index,
                         whiten_filter, output_directory):
    """Run threshold detector and dimensionality reduction using PCA
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    TMP_FOLDER = os.path.join(CONFIG.data.root_folder, output_directory)

    ###################
    # Spike detection #
    ###################

    # FIXME: buffer_size
    bp = BatchProcessor(
        standarized_path,
        standarized_params['dtype'],
        standarized_params['n_channels'],
        standarized_params['data_format'],
        CONFIG.resources.max_memory,
        buffer_size=0)

    path_to_spike_index_clear = os.path.join(TMP_FOLDER,
                                             'spike_index_clear.npy')

    # clear spikes
    if os.path.exists(path_to_spike_index_clear):
        # if it exists, load it...
        logger.info('Found file in {}, loading it...'
                    .format(path_to_spike_index_clear))
        clear = np.load(path_to_spike_index_clear)
    else:
        # if it doesn't, detect spikes...
        logger.info('Did not find file in {}, finding spikes using threshold'
                    ' detector...'.format(path_to_spike_index_clear))

        # apply threshold detector on standarized data
        spikes = bp.multi_channel_apply(
            detect.run,
            mode='memory',
            cleanup_function=detect.fix_indexes,
            neighbors=CONFIG.neighChannels,
            spike_size=CONFIG.spikeSize,
            std_factor=CONFIG.stdFactor)
        clear = np.vstack(spikes)

        logger.info('Removing clear indexes outside the allowed range to '
                    'draw a complete waveform...')
        clear, _ = (detect.remove_incomplete_waveforms(
            clear, CONFIG.spikeSize + CONFIG.templatesMaxShift,
            bp.reader._n_observations))

        logger.info('Saving spikes in {}...'.format(path_to_spike_index_clear))
        np.save(path_to_spike_index_clear, clear)

    path_to_spike_index_collision = os.path.join(TMP_FOLDER,
                                                 'spike_index_collision.npy')

    # collided spikes
    if os.path.exists(path_to_spike_index_collision):
        # if it exists, load it...
        logger.info('Found collided spikes in {}, loading them...'
                    .format(path_to_spike_index_collision))
        collision = np.load(path_to_spike_index_collision)

        if collision.shape[0] != clear.shape[0]:
            raise ValueError('Found collision spike index in {}, '
                             'Since threshold detector is selected,'
                             'all clear spikes are considered collision,'
                             'but number are different. There are {}'
                             'collision spikes and {} clear spikes'
                             .format(path_to_spike_index_collision,
                                     collision.shape[0],
                                     clear.shape[0]))
    else:
        # triage is not implemented on threshold detector, return empty array
        logger.info('Creating empty array for'
                    ' collided spikes (collision detection is not implemented'
                    ' with threshold detector. Saving them in {}'
                    .format(path_to_spike_index_collision))
        collision = clear
        np.save(path_to_spike_index_collision, collision)

    #########################
    # PCA - rotation matrix #
    #########################

    # compute per-batch sufficient statistics for PCA on standarized data
    logger.info('Computing PCA sufficient statistics...')
    stats = bp.multi_channel_apply(
        dim_red.suff_stat,
        mode='memory',
        spike_index=clear,
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
    recordings = RecordingsReader(standarized_path)
    scores = dim_red.score(recordings, rotation,
                           channel_index,
                           clear)

    #################
    # Whiten scores #
    #################
    scores = whiten.score(scores, clear[:, 1], whiten_filter)

    # transform scores to location + shape feature space
    if CONFIG.clustering.clustering_method == 'location':
        scores = get_locations_features_threshold(scores, clear[:, 1],
                                                  channel_index,
                                                  CONFIG.geom)
    # saves score
    path_to_score = os.path.join(TMP_FOLDER, 'score_clear.npy')
    np.save(path_to_score, scores)
    logger.info('Saved spike scores in {}...'.format(path_to_score))

    return scores, clear, collision


def _neural_network_detection(standarized_path, standarized_params,
                              channel_index, whiten_filter, output_directory):
    """Run neural network detection and autoencoder dimensionality reduction
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    TMP_FOLDER = os.path.join(CONFIG.data.root_folder, output_directory)

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

        # Run neural net preprocessor
        # Batch processor
        # FIXME: buffer_size
        bp = BatchProcessor(standarized_path,
                            standarized_params['dtype'],
                            standarized_params['n_channels'],
                            standarized_params['data_format'],
                            CONFIG.resources.max_memory,
                            buffer_size=0)

        # make tensorflow tensors and neural net classes
        detection_th = CONFIG.neural_network_detector.threshold_spike
        triage_th = CONFIG.neural_network_triage.threshold_collision
        detection_fname = CONFIG.neural_network_detector.filename
        ae_fname = CONFIG.neural_network_autoencoder.filename
        triage_fname = CONFIG.neural_network_triage.filename
        (x_tf, output_tf,
         NND, NNT) = neuralnetwork.prepare_nn(channel_index,
                                              whiten_filter,
                                              detection_th,
                                              triage_th,
                                              detection_fname,
                                              ae_fname,
                                              triage_fname)

        # run nn preprocess batch-wsie
        mc = bp.multi_channel_apply
        res = mc(
            neuralnetwork.run_detect_triage_featurize,
            mode='memory',
            cleanup_function=neuralnetwork.fix_indexes,
            x_tf=x_tf,
            output_tf=output_tf,
            NND=NND,
            NNT=NNT)

        # save clear spikes
        clear = np.concatenate([element[1] for element in res], axis=0)
        logger.info('Removing clear indexes outside the allowed range to '
                    'draw a complete waveform...')
        clear, idx = detect.remove_incomplete_waveforms(
            clear, CONFIG.spikeSize + CONFIG.templatesMaxShift,
            bp.reader._n_observations)
        np.save(path_to_spike_index_clear, clear)
        logger.info('Saved spike index clear in {}...'
                    .format(path_to_spike_index_clear))

        # save collided spikes
        collision = np.concatenate([element[2] for element in res], axis=0)
        logger.info('Removing collision indexes outside the allowed range to '
                    'draw a complete waveform...')
        collision, _ = detect.remove_incomplete_waveforms(
            collision, CONFIG.spikeSize + CONFIG.templatesMaxShift,
            bp.reader._n_observations)
        np.save(path_to_spike_index_collision, collision)
        logger.info('Saved spike index collision in {}...'
                    .format(path_to_spike_index_collision))

        # get scores
        scores = np.concatenate([element[0] for element in res], axis=0)
        logger.info(
            'Removing scores for indexes outside the allowed range to '
            'draw a complete waveform...')
        scores = scores[idx]

        # save rotation
        detector_filename = CONFIG.neural_network_detector.filename
        autoencoder_filename = CONFIG.neural_network_autoencoder.filename

        NND = neuralnetwork.NeuralNetDetector(
            detector_filename, autoencoder_filename)
        rotation = NND.load_rotation()
        path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')
        np.save(path_to_rotation, rotation)
        logger.info(
            'Saved rotation matrix in {}...'.format(path_to_rotation))

        # transform scores to location + shape feature space
        if CONFIG.clustering.clustering_method == 'location':
            scores = get_locations_features(scores, rotation, clear[:, 1],
                                            channel_index, CONFIG.geom)
        # saves score
        np.save(path_to_score, scores)
        logger.info('Saved spike scores in {}...'.format(path_to_score))

    return scores, clear, collision


def get_locations_features(scores, rotation, main_channel,
                           channel_index, channel_geometry):

    n_data, n_features, n_neigh = scores.shape

    rot_rot = np.matmul(np.transpose(rotation), rotation)
    reshaped_score = np.reshape(np.transpose(scores, [0, 2, 1]),
                                [n_data*n_neigh, n_features])
    energy = np.sqrt(np.sum(
        np.reshape(np.multiply(np.matmul(reshaped_score, rot_rot),
                               reshaped_score),
                   [n_data, n_neigh, n_features]), 2))

    channel_index_per_data = channel_index[main_channel, :]

    channel_geometry = np.vstack((channel_geometry, np.zeros((1, 2), 'int32')))
    channel_locations_all = channel_geometry[channel_index_per_data]
    xy = np.divide(np.sum(np.multiply(energy[:, :, np.newaxis],
                                      channel_locations_all), axis=1),
                   np.sum(energy, axis=1, keepdims=True))
    scores = np.concatenate((xy, scores[:, :, 0]), 1)

    if scores.shape[0] != n_data:
        raise ValueError('Number of clear spikes changed from {} to {}'
                         .format(n_data, scores.shape[0]))

    if scores.shape[1] != (n_features+channel_geometry.shape[1]):
        raise ValueError('There are {} shape features and {} location features'
                         'but {} features are created'.
                         format(n_features,
                                channel_geometry.shape[1],
                                scores.shape[1]))

    scores = np.divide((scores - np.mean(scores, axis=0, keepdims=True)),
                       np.std(scores, axis=0, keepdims=True))

    return scores[:, :, np.newaxis]


def get_locations_features_threshold(scores, main_channel,
                                     channel_index, channel_geometry):

    n_data, n_features, n_neigh = scores.shape

    energy = np.linalg.norm(scores, axis=1)

    channel_index_per_data = channel_index[main_channel, :]

    channel_geometry = np.vstack((channel_geometry, np.zeros((1, 2), 'int32')))
    channel_locations_all = channel_geometry[channel_index_per_data]
    xy = np.divide(np.sum(np.multiply(energy[:, :, np.newaxis],
                                      channel_locations_all), axis=1),
                   np.sum(energy, axis=1, keepdims=True))
    scores = np.concatenate((xy, scores[:, :, 0]), 1)

    if scores.shape[0] != n_data:
        raise ValueError('Number of clear spikes changed from {} to {}'
                         .format(n_data, scores.shape[0]))

    if scores.shape[1] != (n_features+channel_geometry.shape[1]):
        raise ValueError('There are {} shape features and {} location features'
                         'but {} features are created'
                         .format(n_features,
                                 channel_geometry.shape[1],
                                 scores.shape[1]))

    scores = np.divide((scores - np.mean(scores, axis=0, keepdims=True)),
                       np.std(scores, axis=0, keepdims=True))

    return scores[:, :, np.newaxis]
