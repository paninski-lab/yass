import logging
import os.path
import os
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

import numpy as np

from yass import read_config
from yass.threshold.detect import threshold
from yass.threshold.dimensionality_reduction import pca
from yass.preprocess.batch import whiten
from yass.util import save_numpy_object


def run(standarized_path, standarized_params,
        whiten_filter, if_file_exists,
        save_results, temporal_features=3,
        std_factor=4):
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
                      std_factor,
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
                               temporal_features,
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
