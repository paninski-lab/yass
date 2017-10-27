import os

import pytest

import yass
from yass.preprocessing import Preprocessor
from yass.mainprocess import Mainprocessor
from yass import process
from yass import set_config, reset_config


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
    cfg = yass.Config.from_yaml(path_to_config)

    pp = Preprocessor(cfg)
    score, clr_idx, spt = pp.process()

    mp = Mainprocessor(cfg, score, clr_idx, spt)
    spike_train, spt_left = mp.mainProcess()


def test_process_1k(path_to_config_1k):
    cfg = yass.Config.from_yaml(path_to_config_1k)

    pp = Preprocessor(cfg)
    score, clr_idx, spt = pp.process()

    mp = Mainprocessor(cfg, score, clr_idx, spt)
    spike_train, spt_left = mp.mainProcess()


def test_new_process(path_to_config):
    cfg = yass.Config.from_yaml(path_to_config)

    pp = Preprocessor(cfg)
    score, clr_idx, spt = pp.process()

    set_config(path_to_config)

    spike_train, spt_left, templates = process.run(score, clr_idx, spt)


def test_new_process_shows_error_if_empty_config():
    with pytest.raises(ValueError):
        process.run(None, None, None)
