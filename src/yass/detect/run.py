"""
Detection pipeline
"""
import logging
import os.path

import numpy as np

from yass import read_config
from yass.batch import BatchProcessor, RecordingsReader
from yass.threshold.detect import threshold
from yass.threshold import detect
from yass.threshold import dimensionality_reduction as dim_red
from yass import neuralnetwork
from yass.preprocess import whiten


# TODO: missing parameters docs
def run(standarized_path, standarized_params,
        channel_index, whiten_filter, output_directory='tmp/',
        if_file_exists='skip', save_partial_results=False):
    """Execute detect step

    Parameters
    ----------
    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/

    if_file_exists: str, optional
      One of 'overwrite', 'abort', 'skip'. Control de behavior for every
      generated file. If 'overwrite' it replaces the files if any exist,
      if 'abort' it raises a ValueError exception if any file exists,
      if 'skip' if skips the operation if any file exists

    save_partial_results: bool, optional
        Whether to save partial results to disk, defaults to false

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
    CONFIG.data.root_folder/output_directory/ (if save_partial_results is
    True):

    * ``spike_index_clear.npy`` - Same as spike_index_clear returned
    * ``spike_index_collision.npy`` - Same as spike_index_collision returned
    * ``rotation.npy`` - Rotation matrix for dimensionality reduction
    * ``score_clear.npy`` - Scores for clear spikes

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/detect.py
    """
    CONFIG = read_config()

    # run detection
    if CONFIG.detect.method == 'threshold':
        return run_threshold(standarized_path,
                             standarized_params,
                             channel_index,
                             whiten_filter,
                             output_directory,
                             if_file_exists,
                             save_partial_results)
    elif CONFIG.detect.method == 'nn':
        return run_neural_network(standarized_path,
                                  standarized_params,
                                  channel_index,
                                  whiten_filter,
                                  output_directory,
                                  if_file_exists,
                                  save_partial_results)


def run_threshold(standarized_path, standarized_params, channel_index,
                  whiten_filter, output_directory, if_file_exists,
                  save_partial_results):
    """Run threshold detector and dimensionality reduction using PCA
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    TMP_FOLDER = (os.path.join(CONFIG.data.root_folder, output_directory)
                  if save_partial_results else None)

    ###################
    # Spike detection #
    ###################

    clear = threshold(standarized_path,
                      standarized_params['dtype'],
                      standarized_params['n_channels'],
                      standarized_params['data_format'],
                      CONFIG.resources.max_memory,
                      CONFIG.neigh_channels,
                      CONFIG.spikeSize,
                      CONFIG.spikeSize + CONFIG.templatesMaxShift,
                      CONFIG.stdFactor,
                      TMP_FOLDER,
                      'spike_index_clear.npy',
                      'spike_index_collision.npy',
                      if_file_exists=if_file_exists)

    #######
    # PCA #
    #######

    recordings = RecordingsReader(standarized_path)

    scores, clear, _ = dim_red.pca(standarized_path,
                                   standarized_params['dtype'],
                                   standarized_params['n_channels'],
                                   standarized_params['data_format'],
                                   recordings,
                                   clear,
                                   CONFIG.spikeSize,
                                   CONFIG.detect.temporal_features,
                                   CONFIG.neigh_channels,
                                   channel_index,
                                   CONFIG.resources.max_memory,
                                   output_path=TMP_FOLDER,
                                   save_rotation_matrix='rotation.npy',
                                   save_scores='score_clear.npy',
                                   if_file_exists=if_file_exists)

    #################
    # Whiten scores #
    #################

    scores = whiten.score(scores, clear[:, 1], whiten_filter)

    # TODO: this shouldn't be here
    # transform scores to location + shape feature space
    if CONFIG.cluster.method == 'location':
        scores = get_locations_features_threshold(scores, clear[:, 1],
                                                  channel_index,
                                                  CONFIG.geom)
    # saves score
    if TMP_FOLDER:
        path_to_score = os.path.join(TMP_FOLDER, 'score_clear.npy')
        np.save(path_to_score, scores)
        logger.info('Saved spike scores in {}...'.format(path_to_score))

    return scores, clear, np.copy(clear)


def run_neural_network(standarized_path, standarized_params,
                       channel_index, whiten_filter, output_directory,
                       if_file_exists, save_partial_results):
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
        bp = BatchProcessor(standarized_path,
                            standarized_params['dtype'],
                            standarized_params['n_channels'],
                            standarized_params['data_format'],
                            CONFIG.resources.max_memory,
                            buffer_size=CONFIG.spikeSize)

        # make tensorflow tensors and neural net classes
        detection_th = CONFIG.detect.neural_network_detector.threshold_spike
        triage_th = CONFIG.detect.neural_network_triage.threshold_collision
        detection_fname = CONFIG.detect.neural_network_detector.filename
        ae_fname = CONFIG.detect.neural_network_autoencoder.filename
        triage_fname = CONFIG.detect.neural_network_triage.filename
        (x_tf, output_tf, NND,
         NNAE, NNT) = neuralnetwork.prepare_nn(channel_index,
                                               whiten_filter,
                                               detection_th,
                                               triage_th,
                                               detection_fname,
                                               ae_fname,
                                               triage_fname)

        # run nn preprocess batch-wsie
        # run nn preprocess batch-wsie
        mc = bp.multi_channel_apply
        res = mc(
            neuralnetwork.run_detect_triage_featurize,
            mode='memory',
            cleanup_function=neuralnetwork.fix_indexes,
            x_tf=x_tf,
            output_tf=output_tf,
            NND=NND,
            NNAE=NNAE,
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
        rotation = NNAE.load_rotation()
        path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')
        np.save(path_to_rotation, rotation)
        logger.info(
            'Saved rotation matrix in {}...'.format(path_to_rotation))

        # transform scores to location + shape feature space
        if CONFIG.cluster.method == 'location':
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
