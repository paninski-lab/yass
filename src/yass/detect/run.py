"""
Detection pipeline
"""
import logging
import os.path
import os
from functools import reduce
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

import numpy as np
import tensorflow as tf

from yass import read_config
from yass.batch import BatchProcessor
from yass.threshold.detect import threshold
from yass.threshold import detect
from yass.threshold.dimensionality_reduction import pca
from yass.neuralnetwork import NeuralNetDetector, AutoEncoder, KerasModel
from yass.neuralnetwork.apply import post_processing, fix_indexes_spike_index
from yass.preprocess.batch import whiten
from yass.geometry import n_steps_neigh_channels
from yass.util import file_loader, save_numpy_object, running_on_gpu


def run(standarized_path, standarized_params, whiten_filter,
        if_file_exists='skip', save_results=False):
    """Execute detect step

    Parameters
    ----------
    standarized_path: str or pathlib.Path
        Path to standarized data binary file

    standarized_params: dict, str or pathlib.Path
        Dictionary with standarized data parameters or path to a yaml file

    whiten_filter: numpy.ndarray, str or pathlib.Path
        Whiten matrix or path to a npy file

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

    spike_index_all: numpy.ndarray (n_collided_spikes, 2)
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
    whiten_filter = file_loader(whiten_filter)

    # run detection
    if CONFIG.detect.method == 'threshold':
        return run_threshold(standarized_path,
                             standarized_params,
                             whiten_filter,
                             if_file_exists,
                             save_results)
    elif CONFIG.detect.method == 'nn':
        return run_neural_network(standarized_path,
                                  standarized_params,
                                  whiten_filter,
                                  if_file_exists,
                                  save_results)


def run_threshold(standarized_path, standarized_params,
                  whiten_filter, if_file_exists,
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

    folder = Path(CONFIG.path_to_output_directory, 'detect')
    folder.mkdir(exist_ok=True)

    TMP_FOLDER = str(folder)

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

    # run PCA, save rotation matrix and pca scores under TMP_FOLDER
    # TODO: remove clear as input for PCA and create an independent function
    pca_scores, clear, _ = pca(standarized_path,
                               standarized_params['dtype'],
                               standarized_params['n_channels'],
                               standarized_params['data_order'],
                               clear,
                               CONFIG.spike_size,
                               CONFIG.detect.temporal_features,
                               CONFIG.neigh_channels,
                               CONFIG.channel_index,
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
    scores = whiten.score(pca_scores, clear[:, 1], whiten_filter)

    if save_results:
        # save spike_index_all (same as spike_index_clear for threshold
        # detector)
        path_to_spike_index_all = os.path.join(TMP_FOLDER,
                                               filename_spike_index_all)
        save_numpy_object(clear, path_to_spike_index_all, if_file_exists,
                          name='Spike index all')

    # FIXME: always saving scores since they are loaded by the clustering
    # step, we need to find a better way to do this, since the current
    # clustering code is going away soon this is a tmp solution
    # saves scores
    # saves whiten scores
    path_to_scores = os.path.join(TMP_FOLDER, filename_scores_clear)
    save_numpy_object(scores, path_to_scores, if_file_exists,
                      name='scores')

    return clear, np.copy(clear)


def run_neural_network(standarized_path, standarized_params,
                       whiten_filter,
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

    folder = Path(CONFIG.path_to_output_directory, 'detect')
    folder.mkdir(exist_ok=True)
    TMP_FOLDER = str(folder)

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
        max_memory = (CONFIG.resources.max_memory_gpu if running_on_gpu() else
                      CONFIG.resources.max_memory)

        # instantiate batch processor
        bp = BatchProcessor(standarized_path,
                            standarized_params['dtype'],
                            standarized_params['n_channels'],
                            standarized_params['data_order'],
                            max_memory,
                            buffer_size=CONFIG.spike_size)

        # load parameters
        detection_th = CONFIG.detect.neural_network_detector.threshold_spike
        triage_th = CONFIG.detect.neural_network_triage.threshold_collision
        detection_fname = CONFIG.detect.neural_network_detector.filename
        ae_fname = CONFIG.detect.neural_network_autoencoder.filename
        triage_fname = CONFIG.detect.neural_network_triage.filename

        # instantiate neural networks
        NND = NeuralNetDetector.load(detection_fname, detection_th,
                                     CONFIG.channel_index)
        triage = KerasModel(triage_fname,
                            allow_longer_waveform_length=True,
                            allow_more_channels=True)
        NNAE = AutoEncoder.load(ae_fname, input_tensor=NND.waveform_tf)

        neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)

        fn = fix_indexes_spike_index

        # detector
        with tf.Session() as sess:
            # get values of above tensors
            NND.restore(sess)

            res = bp.multi_channel_apply(NND.predict_recording,
                                         mode='memory',
                                         sess=sess,
                                         output_names=('spike_index',
                                                       'waveform'),
                                         cleanup_function=fn)

        spikes_all, wfs = zip(*res)

        spikes_all = np.concatenate(spikes_all, axis=0)
        wfs = np.concatenate(wfs, axis=0)

        idx_clean = triage.predict_with_threshold(x=wfs,
                                                  threshold=triage_th)
        score = NNAE.predict(wfs)
        rot = NNAE.load_rotation()
        neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)

        (scores, clear) = post_processing(score,
                                          spikes_all,
                                          idx_clean,
                                          rot,
                                          neighbors)

        # get clear spikes
        logger.info('Removing clear indexes outside the allowed range to '
                    'draw a complete waveform...')
        clear, idx = detect.remove_incomplete_waveforms(
            clear, CONFIG.spike_size + CONFIG.templates.max_shift,
            bp.reader._n_observations)

        # get all spikes
        logger.info('Removing indexes outside the allowed range to '
                    'draw a complete waveform...')
        spikes_all, _ = detect.remove_incomplete_waveforms(
            spikes_all, CONFIG.spike_size + CONFIG.templates.max_shift,
            bp.reader._n_observations)

        # get scores
        logger.info(
            'Removing scores for indexes outside the allowed range to '
            'draw a complete waveform...')
        scores = scores[idx]

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
            np.save(path_to_rotation, rot)
            logger.info('Saved rotation matrix in {}...'
                        .format(path_to_rotation))

        # FIXME: always saving scores since they are loaded by the clustering
        # step, we need to find a better way to do this, since the current
        # clustering code is going away soon this is a tmp solution
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
        logger.warning('Skipped execution. All output files exist'
                       ', loading them...')
        scores = np.load(path_to_score)
        clear = np.load(path_to_spike_index_clear)
        spikes_all = np.load(path_to_spike_index_all)

    else:
        raise ValueError('Invalid value for if_file_exists {}'
                         'must be one of overwrite, abort or skip'
                         .format(if_file_exists))

    return clear, spikes_all
