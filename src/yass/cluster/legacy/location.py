import pkg_resources
from os.path import join
import logging
import datetime
import numpy as np

from yass import read_config
from yass.cluster.legacy.subsample import random_subsample
from yass.cluster.legacy.triage import triage
from yass.cluster.legacy.util import (calculate_sparse_rhat,
                                      run_cluster_location)
from yass.neuralnetwork import AutoEncoder
from yass.mfm import get_core_data


def location(spike_index, detect_method='threshold',
             autoencoder='ae_nn1.ckpt'):
    """Spike clustering

    Parameters
    ----------
    spike_index: numpy.ndarray (n_clear_spikes, 2), str or Path
        2D array with indexes for spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum). Or path to an npy file

    Returns
    -------
    spike_train
    """
    autoencoder = expand_asset_model(autoencoder)

    CONFIG = read_config()

    # NOTE: this is not the right way to set defaults, the function should
    # list all parameters needed and provide defaults for them, since the
    # current code looks for parameters inside CONFIG, we are doing it like
    # this, we will remove this clustering method so de are not refactoring
    # this, new clustering methods should list all parameters in the function
    # signature
    defaults = {
        'method': 'location',
        'save_results': False,
        'masking_threshold': [0.9, 0.5],
        'n_split': 5,
        'max_n_spikes': 10000,
        'min_spikes': 0,
        'prior': {
            'beta': 1,
            'a': 1,
            'lambda0': 0.01,
            'nu': 5,
            'V': 2,
        },
        'coreset': {
            'clusters': 10,
            'threshold': 0.95,
        },
        'triage': {
            'nearest_neighbors': 20,
            'percent': 0.1,
        }
    }

    CONFIG._set_param('cluster', defaults)

    # load files in case they are strings or Path objects
    path_to_scores = join(CONFIG.path_to_output_directory,
                          'detect', 'scores_clear.npy')
    scores = np.load(path_to_scores)

    startTime = datetime.datetime.now()

    if detect_method == 'threshold':
        scores = get_locations_features_threshold(scores, spike_index[:, 1],
                                                  CONFIG.channel_index,
                                                  CONFIG.geom)
    else:
        autoencoder = AutoEncoder.load(autoencoder)
        rotation = autoencoder.load_rotation()
        threshold = 2
        scores = get_locations_features_nnet(scores, rotation,
                                             spike_index[:, 1],
                                             CONFIG.channel_index, CONFIG.geom,
                                             threshold)
        idx_nan = np.where(np.isnan(np.sum(scores, axis=(1, 2))))[0]
        scores = np.delete(scores, idx_nan, 0)
        spike_index = np.delete(spike_index, idx_nan, 0)

    scores_all = np.copy(scores)
    spike_index_all = np.copy(spike_index)

    Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

    logger = logging.getLogger(__name__)

    ##########
    # Triage #
    ##########

    _b = datetime.datetime.now()
    logger.info("Randomly subsampling...")
    scores, spike_index = random_subsample(scores, spike_index,
                                           CONFIG.cluster.max_n_spikes)
    logger.info("Triaging...")
    scores, spike_index = triage(scores, spike_index,
                                 CONFIG.cluster.triage.nearest_neighbors,
                                 CONFIG.cluster.triage.percent,
                                 CONFIG.cluster.method == 'location')
    Time['t'] += (datetime.datetime.now()-_b).total_seconds()

    _b = datetime.datetime.now()
    logger.info("Clustering...")
    vbParam, tmp_loc, scores, spike_index = run_cluster_location(
        scores, spike_index, CONFIG.cluster.min_spikes, CONFIG)
    Time['s'] += (datetime.datetime.now()-_b).total_seconds()

    vbParam.rhat = calculate_sparse_rhat(vbParam, tmp_loc, scores_all,
                                         spike_index_all,
                                         CONFIG.neigh_channels)
    idx_keep = get_core_data(vbParam, scores_all, np.inf, 2)
    spike_train = vbParam.rhat[idx_keep]
    spike_train[:, 0] = spike_index_all[spike_train[:, 0].astype('int32'), 0]

    # report timing
    currentTime = datetime.datetime.now()
    logger.info("Mainprocess done in {0} seconds.".format(
        (currentTime - startTime).seconds))
    logger.info("\ttriage:\t{0} seconds".format(Time['t']))
    logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
    logger.info("\tmasking:\t{0} seconds".format(Time['m']))
    logger.info("\tclustering:\t{0} seconds".format(Time['s']))

    return spike_train, tmp_loc, vbParam


def get_locations_features_nnet(scores, rotation, main_channel,
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
