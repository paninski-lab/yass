"""
neuralnetwork module tests
"""
import os.path as path

import numpy as np
import yaml

import yass
from yass.batch import RecordingsReader, BatchProcessor
from yass import neuralnetwork
from yass.neuralnetwork import nn_detection as nn


def test_can_use_neural_network_detector(path_to_tests):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()
    data = RecordingsReader(path.join(path_to_tests,
                                      'data/standarized.bin'),
                            mmap=False).data.T
    nn(data, CONFIG.neighChannels, CONFIG.geom,
       CONFIG.spikes.temporal_features, 3,
       CONFIG.neural_network_detector.threshold_spike,
       CONFIG.neural_network_triage.threshold_collision,
       CONFIG.neural_network_detector.filename,
       CONFIG.neural_network_autoencoder.filename,
       CONFIG.neural_network_triage.filename)


def test_splitting_in_batches_does_not_affect_result(path_to_tests):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    PATH_TO_DATA = path.join(path_to_tests, 'data/standarized.bin')

    data = RecordingsReader(PATH_TO_DATA, mmap=False).data.T

    with open(path.join(path_to_tests, 'data/standarized.yaml')) as f:
        PARAMS = yaml.load(f)

    # buffer size makes sure we can detect spikes if they appear at the end of
    # any batch
    bp = BatchProcessor(PATH_TO_DATA, PARAMS['dtype'], PARAMS['n_channels'],
                        PARAMS['data_format'], '100MB', buffer_size=15)
    mc = bp.multi_channel_apply
    res = mc(nn,
             mode='memory',
             cleanup_function=neuralnetwork.fix_indexes,
             neighbors=CONFIG.neighChannels,
             geom=CONFIG.geom,
             temporal_features=CONFIG.spikes.temporal_features,
             temporal_window=3,
             th_detect=CONFIG.neural_network_detector.threshold_spike,
             th_triage=CONFIG.neural_network_triage.threshold_collision,
             detector_filename=CONFIG.neural_network_detector.filename,
             autoencoder_filename=CONFIG.neural_network_autoencoder.filename,
             triage_filename=CONFIG.neural_network_triage.filename)

    scores_batch = np.concatenate([element[0] for element in res], axis=0)
    clear_batch = np.concatenate([element[1] for element in res], axis=0)
    collision_batch = np.concatenate([element[2] for element in res], axis=0)

    (scores, clear,
     collision) = nn(data, CONFIG.neighChannels, CONFIG.geom,
                     CONFIG.spikes.temporal_features, 3,
                     CONFIG.neural_network_detector.threshold_spike,
                     CONFIG.neural_network_triage.threshold_collision,
                     CONFIG.neural_network_detector.filename,
                     CONFIG.neural_network_autoencoder.filename,
                     CONFIG.neural_network_triage.filename)

    np.testing.assert_array_equal(clear_batch, clear)
    np.testing.assert_array_equal(collision_batch, collision)
    np.testing.assert_array_equal(scores_batch, scores)


def test_can_train_nnet(path_to_tests):
    pass
