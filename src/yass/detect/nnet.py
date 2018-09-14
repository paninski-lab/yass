import pkg_resources
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
from yass.threshold import detect
from yass.neuralnetwork import NeuralNetDetector, AutoEncoder, KerasModel
from yass.neuralnetwork.apply import post_processing, fix_indexes_spike_index
from yass.geometry import n_steps_neigh_channels
from yass.util import running_on_gpu


def run(standarized_path, standarized_params, whiten_filter, if_file_exists,
        save_results, detector='detect_nn1.ckpt',
        detector_threshold=0.5,
        triage='triage-31wf7ch-15-Aug-2018@00-17-16.h5',
        triage_threshold=0.5,
        autoencoder='ae_nn1.ckpt'):
    """Run neural network detection and autoencoder dimensionality reduction

    Returns
    -------
    scores
      Scores for all spikes

    spike_index_clear
      Spike indexes for clear spikes

    spike_index_all
      Spike indexes for all spikes

    detector:
        model name, can be any of the models included in yass
        (detectnet1.ckpt),
        a relative folder to data.root_fodler (e.g.
        $ROOT_FOLDER/models/mymodel.ckpt) or an absolute path to a model
        (e.g. /path/to/my/model.ckpt). In the same folder as your model, there
        must be a yaml file with the number and size of the filters, the file
        should be named exactly as your model but with yaml extension
        see yass/src/assets/models/ for an example
    """
    logger = logging.getLogger(__name__)

    detector = expand_asset_model(detector)
    triage = expand_asset_model(triage)
    autoencoder = expand_asset_model(autoencoder)

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

        # instantiate neural networks
        NND = NeuralNetDetector.load(detector, detector_threshold,
                                     CONFIG.channel_index)
        triage = KerasModel(triage,
                            allow_longer_waveform_length=True,
                            allow_more_channels=True)
        NNAE = AutoEncoder.load(autoencoder, input_tensor=NND.waveform_tf)

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
                                                  threshold=triage_threshold)
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


def expand_asset_model(value):
    """Expand filenames
    """
    # if absolute path, just return the value
    if value.startswith('/'):
        new_value = value

    # else, look into assets
    else:
        path = 'assets/models/{}'.format(value)
        new_value = pkg_resources.resource_filename('yass', path)

    return new_value
