from os.path import join
import logging
import datetime
import numpy as np

from yass import read_config
from yass.util import file_loader, check_for_files, LoadFile
from yass.cluster.subsample import random_subsample
from yass.cluster.triage import triage
from yass.cluster.coreset import coreset
from yass.cluster.mask import getmask
from yass.cluster.util import (run_cluster, calculate_sparse_rhat,
                               run_cluster_location)
from yass.neuralnetwork import AutoEncoder
from yass.mfm import get_core_data


@check_for_files(filenames=[LoadFile(join('cluster',
                                          'spike_train_cluster.npy')),
                            LoadFile(join('cluster', 'tmp_loc.npy')),
                            LoadFile(join('cluster', 'vbPar.pickle'))],
                 mode='values', relative_to=None,
                 auto_save=True)
def run(spike_index, if_file_exists='skip', save_results=False):
    """Spike clustering

    Parameters
    ----------
    scores: numpy.ndarray (n_spikes, n_features, n_channels), str or Path
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels. Or path to a npy file

    spike_index: numpy.ndarray (n_clear_spikes, 2), str or Path
        2D array with indexes for spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum). Or path to an npy file

    output_directory: str, optional
        Location to store/look for the generate spike train, relative to
        CONFIG.data.root_folder

    if_file_exists: str, optional
      One of 'overwrite', 'abort', 'skip'. Control de behavior for the
      spike_train_cluster.npy. file If 'overwrite' it replaces the files if
      exists, if 'abort' it raises a ValueError exception if exists,
      if 'skip' it skips the operation if the file exists (and returns the
      stored file)

    save_results: bool, optional
        Whether to save spike train to disk
        (in CONFIG.data.root_folder/relative_to/spike_train_cluster.npy),
        defaults to False

    Returns
    -------
    spike_train: (TODO add documentation)

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/cluster.py

    """
    spike_index = file_loader(spike_index)

    CONFIG = read_config()

    # load files in case they are strings or Path objects
    path_to_scores = join(CONFIG.path_to_output_directory,
                          'detect', 'scores_clear.npy')
    scores = np.load(path_to_scores)

    startTime = datetime.datetime.now()

    if CONFIG.detect.method == 'threshold':
        scores = get_locations_features_threshold(scores, spike_index[:, 1],
                                                  CONFIG.channel_index,
                                                  CONFIG.geom)
    else:
        ae_fname = CONFIG.detect.neural_network_autoencoder.filename
        autoencoder = AutoEncoder.load(ae_fname)
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

    # ###########
    # # Coreset #
    # ###########
    # _b = datetime.datetime.now()
    # logger.info("Coresetting...")
    # groups = coreset(scores,
    #                  spike_index,
    #                  CONFIG.cluster.coreset.clusters,
    #                  CONFIG.cluster.coreset.threshold)
    # Time['c'] += (datetime.datetime.now() - _b).total_seconds()

    # ###########
    # # Masking #
    # ###########
    # _b = datetime.datetime.now()
    # logger.info("Masking...")
    # masks = getmask(scores, spike_index, groups,
    #                 CONFIG.cluster.masking_threshold)
    # Time['m'] += (datetime.datetime.now() - _b).total_seconds()

    # ##############
    # # Clustering #
    # ##############
    # _b = datetime.datetime.now()
    # logger.info("Clustering...")
    # vbParam, tmp_loc, scores, spike_index = run_cluster(
    #     scores, masks, groups, spike_index,
    #     CONFIG.cluster.min_spikes, CONFIG)
    # Time['s'] += (datetime.datetime.now()-_b).total_seconds()

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
