"""
Testing neural network training
"""
from os import path
import numpy as np
from yass.config import Config
from yass.util import load_yaml
from yass import neuralnetwork

PATH_TO_REF = '/home/Edu/data/nnet'


def test_can_generate_training_data():
    pass


def test_can_train_neural_networks(path_to_tests):
    path_to_spike_train = path.join(PATH_TO_REF, 'spike_train.npy')
    path_to_config = path.join(PATH_TO_REF, 'config.yaml')
    path_to_config_train = path.join(path_to_tests, 'config_train.yaml')

    spike_train = np.load(path_to_spike_train)
    CONFIG = Config.from_yaml(path_to_config)
    CONFIG_TRAIN = load_yaml(path_to_config_train)

    neuralnetwork.train(CONFIG, CONFIG_TRAIN, spike_train,
                        data_folder=PATH_TO_REF)
