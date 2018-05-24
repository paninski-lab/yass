"""
Detection pipeline
"""
import logging
import os.path
from functools import reduce

import numpy as np

from yass import read_config, GPU_ENABLED
from yass.batch import BatchProcessor, RecordingsReader
from yass.threshold.detect import threshold
from yass.threshold import detect
from yass.threshold.dimensionality_reduction import pca
from yass import neuralnetwork
from yass.preprocess import whiten
from yass.geometry import n_steps_neigh_channels
from yass.util import file_loader, save_numpy_object


def run(standarized_path, standarized_params,
        channel_index, whiten_filter, output_directory='tmp/',
        if_file_exists='skip', save_results=False):
    """Execute detect step

    Parameters
    ----------
    standarized_path: str or pathlib.Path
        Path to standarized data binary file

    standarized_params: dict, str or pathlib.Path
        Dictionary with standarized data parameters or path to a yaml file

    channel_index: numpy.ndarray, str or pathlib.Path
        Channel index or path to a npy file

    whiten_filter: numpy.ndarray, str or pathlib.Path
        Whiten matrix or path to a npy file

    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/

    if_file_exists: str, optional
      One of 'overwrite', 'abort', 'skip'. Control de behavior for every
      generated file. If 'overwrite' it replaces the files if any exist,
      if 'abort' it raises a ValueError exception if any file exists,
      if 'skip' if skips the operation if any file exists

    save_results: bool, optional
        Whether to save results to disk, defaults to False

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

    spike_index_call: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for all spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/ (if save_results is
    True):

    * ``spike_index_clear.npy`` - Same as spike_index_clear returned
    * ``spike_index_all.npy`` - Same as spike_index_collision returned
    * ``rotation.npy`` - Rotation matrix for dimensionality reduction
    * ``scores_clear.npy`` - Scores for clear spikes

    Threshold detector runs on CPU, neural network detector runs CPU and GPU,
    depending on how tensorflow is configured.

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/detect.py
    """
    CONFIG = read_config()

    # load files in case they are strings or Path objects
    standarized_params = file_loader(standarized_params)
    channel_index = file_loader(channel_index)
    whiten_filter = file_loader(whiten_filter)

    # run detection
    if CONFIG.detect.method == 'threshold':
        return run_threshold(standarized_path,
                             standarized_params,
                             channel_index,
                             whiten_filter,
                             output_directory,
                             if_file_exists,
                             save_results)
    elif CONFIG.detect.method == 'nn':
        return run_neural_network(standarized_path,
                                  standarized_params,
                                  channel_index,
                                  whiten_filter,
                                  output_directory,
                                  if_file_exists,
                                  save_results)


def run_threshold(standarized_path, standarized_params, channel_index,
                  whiten_filter, output_directory, if_file_exists,
                  save_results):
    """Run threshold detector and dimensionality reduction using PCA


    Returns
    -------
    scores
      Scores for all spikes

    spike_index_clear
      Spike indexes for clear spikes

    spike_index_all
      Spike indexes for all spikes
    """
    logger = logging.getLogger(__name__)

    logger.debug('Running threshold detector...')

    CONFIG = read_config()

    # Set TMP_FOLDER to None if not save_results, this will disable
    # saving results in every function below
    TMP_FOLDER = (os.path.join(CONFIG.data.root_folder, output_directory)
                  if save_results else None)

    # files that will be saved if enable by the if_file_exists option
    filename_index_clear = 'spike_index_clear.npy'
    filename_index_clear_pca = 'spike_index_clear_pca.npy'
    filename_scores_clear = 'scores_clear.npy'
    filename_spike_index_all = 'spike_index_all.npy'
    filename_rotation = 'rotation.npy'

    ###################
    # Spike detection #
    ###################

    # run threshold detection, save clear indexes in TMP/filename_index_clear
    clear = threshold(standarized_path,
                      standarized_params['dtype'],
                      standarized_params['n_channels'],
                      standarized_params['data_order'],
                      CONFIG.resources.max_memory,
                      CONFIG.neigh_channels,
                      CONFIG.spike_size,
                      CONFIG.spike_size + CONFIG.templates.max_shift,
                      CONFIG.detect.threshold_detector.std_factor,
                      TMP_FOLDER,
                      spike_index_clear_filename=filename_index_clear,
                      if_file_exists=if_file_exists)

    #######
    # PCA #
    #######

    recordings = RecordingsReader(standarized_path)

    # run PCA, save rotation matrix and pca scores under TMP_FOLDER
    # TODO: remove clear as input for PCA and create an independent function
    pca_scores, clear, _ = pca(standarized_path,
                               standarized_params['dtype'],
                               standarized_params['n_channels'],
                               standarized_params['data_order'],
                               recordings,
                               clear,
                               CONFIG.spike_size,
                               CONFIG.detect.temporal_features,
                               CONFIG.neigh_channels,
                               channel_index,
                               CONFIG.resources.max_memory,
                               TMP_FOLDER,
                               'scores_pca.npy',
                               filename_rotation,
                               filename_index_clear_pca,
                               if_file_exists)

    #################
    # Whiten scores #
    #################

    # apply whitening to scores
    scores_clear = whiten.score(pca_scores, clear[:, 1], whiten_filter)

    if TMP_FOLDER is not None:
        # saves whiten scores
        path_to_scores = os.path.join(TMP_FOLDER, filename_scores_clear)
        save_numpy_object(scores_clear, path_to_scores, if_file_exists,
                          name='scores')

        # save spike_index_all (same as spike_index_clear for threshold
        # detector)
        path_to_spike_index_all = os.path.join(TMP_FOLDER,
                                               filename_spike_index_all)
        save_numpy_object(clear, path_to_spike_index_all, if_file_exists,
                          name='Spike index all')

    # TODO: this shouldn't be here
    # transform scores to location + shape feature space
    if CONFIG.cluster.method == 'location':
        scores = get_locations_features_threshold(scores_clear, clear[:, 1],
                                                  channel_index,
                                                  CONFIG.geom)

    return scores, clear, np.copy(clear)


def run_neural_network(standarized_path, standarized_params,
                       channel_index, whiten_filter, output_directory,
                       if_file_exists, save_results):
    """Run neural network detection and autoencoder dimensionality reduction

    Returns
    -------
    scores
      Scores for all spikes

    spike_index_clear
      Spike indexes for clear spikes

    spike_index_all
      Spike indexes for all spikes
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    TMP_FOLDER = os.path.join(CONFIG.data.root_folder, output_directory)

    # check if all scores, clear and collision spikes exist..
    path_to_score = os.path.join(TMP_FOLDER, 'scores_clear.npy')
    path_to_spike_index_clear = os.path.join(TMP_FOLDER,
                                             'spike_index_clear.npy')
    path_to_spike_index_all = os.path.join(TMP_FOLDER, 'spike_index_all.npy')
    path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')

    paths = [path_to_score, path_to_spike_index_clear, path_to_spike_index_all]
    exists = [os.path.exists(p) for p in paths]

    if (if_file_exists == 'overwrite' or
        if_file_exists == 'abort' and not any(exists)
       or if_file_exists == 'skip' and not all(exists)):
        max_memory = (CONFIG.resources.max_memory_gpu if GPU_ENABLED else
                      CONFIG.resources.max_memory)

        # Batch processor
        bp = BatchProcessor(standarized_path,
                            standarized_params['dtype'],
                            standarized_params['n_channels'],
                            standarized_params['data_order'],
                            max_memory,
                            buffer_size=CONFIG.spike_size)

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
        neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)
        mc = bp.multi_channel_apply
        res = mc(
            neuralnetwork.run_detect_triage_featurize,
            mode='memory',
            cleanup_function=neuralnetwork.fix_indexes,
            x_tf=x_tf,
            output_tf=output_tf,
            NND=NND,
            NNAE=NNAE,
            NNT=NNT,
            neighbors=neighbors)

        # get clear spikes
        clear = np.concatenate([element[1] for element in res], axis=0)
        logger.info('Removing clear indexes outside the allowed range to '
                    'draw a complete waveform...')
        clear, idx = detect.remove_incomplete_waveforms(
            clear, CONFIG.spike_size + CONFIG.templates.max_shift,
            bp.reader._n_observations)

        # get all spikes
        spikes_all = np.concatenate([element[2] for element in res], axis=0)
        logger.info('Removing indexes outside the allowed range to '
                    'draw a complete waveform...')
        spikes_all, _ = detect.remove_incomplete_waveforms(
            spikes_all, CONFIG.spike_size + CONFIG.templates.max_shift,
            bp.reader._n_observations)

        # get scores
        scores = np.concatenate([element[0] for element in res], axis=0)
        logger.info(
            'Removing scores for indexes outside the allowed range to '
            'draw a complete waveform...')
        scores = scores[idx]

        # transform scores to location + shape feature space
        # TODO: move this to another place
        rotation = NNAE.load_rotation()

        if CONFIG.cluster.method == 'location':
            threshold = 2
            scores = get_locations_features(scores, rotation, clear[:, 1],
                                            channel_index, CONFIG.geom,
                                            threshold)
            idx_nan = np.where(np.isnan(np.sum(scores, axis=(1, 2))))[0]
            scores = np.delete(scores, idx_nan, 0)
            clear = np.delete(clear, idx_nan, 0)

        # save partial results if required
        if save_results:
            # save clear spikes
            np.save(path_to_spike_index_clear, clear)
            logger.info('Saved spike index clear in {}...'
                        .format(path_to_spike_index_clear))

            # save all ppikes
            np.save(path_to_spike_index_all, spikes_all)
            logger.info('Saved spike index all in {}...'
                        .format(path_to_spike_index_all))

            # save rotation
            np.save(path_to_rotation, rotation)
            logger.info('Saved rotation matrix in {}...'
                        .format(path_to_rotation))

            # saves scores
            np.save(path_to_score, scores)
            logger.info('Saved spike scores in {}...'.format(path_to_score))

    elif if_file_exists == 'abort' and any(exists):
        conflict = [p for p, e in zip(paths, exists) if e]
        message = reduce(lambda x, y: str(x)+', '+str(y), conflict)
        raise ValueError('if_file_exists was set to abort, the '
                         'program halted since the following files '
                         'already exist: {}'.format(message))
    elif if_file_exists == 'skip' and all(exists):
        logger.info('Skipped execution. All necessary files exist'
                    ', loading them...')
        scores = np.load(path_to_score)
        clear = np.load(path_to_spike_index_clear)
        spikes_all = np.load(path_to_spike_index_all)

    else:
        raise ValueError('Invalid value for if_file_exists {}'
                         'must be one of overwrite, abort or skip'
                         .format(if_file_exists))

    return scores, clear, spikes_all


def get_locations_features(scores, rotation, main_channel,
                           channel_index, channel_geometry,
                           threshold):

    n_data, n_features, n_neigh = scores.shape

    reshaped_score = np.reshape(np.transpose(scores, [0, 2, 1]),
                                [n_data*n_neigh, n_features])
    energy = np.reshape(np.ptp(np.matmul(
        reshaped_score, rotation.T), 1), (n_data, n_neigh))

    energy = np.piecewise(energy, [energy < threshold,
                                   energy >= threshold],
                          [0, lambda x:x-threshold])

    channel_index_per_data = channel_index[main_channel, :]
    channel_geometry = np.vstack((channel_geometry, np.zeros((1, 2), 'int32')))
    channel_locations_all = channel_geometry[channel_index_per_data]

    xy = np.divide(np.sum(np.multiply(energy[:, :, np.newaxis],
                                      channel_locations_all), axis=1),
                   np.sum(energy, axis=1, keepdims=True))
    noise = np.random.randn(xy.shape[0], xy.shape[1])*(0.00001)
    xy += noise

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
