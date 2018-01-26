"""
neuralnetwork module tests
"""
import os

import yass
from yass.batch import RecordingsReader
from yass.neuralnetwork import nn_detection as nn


def test_can_use_neural_network_detector(path_to_tests):
    yass.set_config(os.path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()
    data = RecordingsReader(os.path.join(path_to_tests,
                                         'data/standarized.bin'),
                            mmap=False).data.T
    nn(data, CONFIG.neighChannels, CONFIG.geom,
       CONFIG.spikes.temporal_features, 3,
       CONFIG.neural_network_detector.threshold_spike,
       CONFIG.neural_network_triage.threshold_collision,
       CONFIG.neural_network_detector.filename,
       CONFIG.neural_network_autoencoder.filename,
       CONFIG.neural_network_triage.filename)


def test_can_train_nnet(path_to_tests):
    pass
