"""
neuralnetwork module tests
"""

import os

import pytest


@pytest.fixture
def path_to_config():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold.yaml')
    return path


def test_can_restore_nnet(path_to_config):
    pass


def test_can_train_nnet(path_to_config):
    pass
