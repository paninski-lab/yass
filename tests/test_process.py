import os

import pytest

import yass
from yass import preprocess
from yass import process
from yass import reset_config


def teardown_function(function):
    reset_config()


@pytest.fixture
def path_to_config():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold.yaml')
    return path


@pytest.fixture
def path_to_config_1k():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold_1k.yaml')
    return path


def test_process(path_to_config):
    yass.set_config(path_to_config)

    score, spike_index_clear, spike_index_collision = preprocess.run()

    (spike_train_clear, templates,
     spike_index_collision) = process.run(score, spike_index_clear,
                                          spike_index_collision)


def test_new_process_shows_error_if_empty_config():
    with pytest.raises(ValueError):
        process.run(None, None, None)
