import os
import pytest
import yass


def test_can_initialize_threshold_config(path_to_threshold_config):
    yass.Config.from_yaml(path_to_threshold_config)


def test_can_initialize_nnet_config(path_to_nnet_config):
    yass.Config.from_yaml(path_to_nnet_config)


def test_can_initialize_config_sample(path_to_config_sample):
    yass.Config.from_yaml(path_to_config_sample)


def test_can_initialize_config_examples_sample(path_to_examples):
    old = os.getcwd()
    os.chdir(path_to_examples)
    yass.Config.from_yaml('config_sample.yaml')
    os.chdir(old)


def test_can_initialize_config_examples_sample_complete(path_to_examples):
    old = os.getcwd()
    os.chdir(path_to_examples)
    yass.Config.from_yaml('config_sample_complete.yaml')
    os.chdir(old)


def test_cannot_modify_once_initialized(path_to_threshold_config):
    cfg = yass.Config.from_yaml(path_to_threshold_config)

    with pytest.raises(AttributeError):
        cfg.param = 1


def test_extra_parameters_are_computed(path_to_threshold_config):
    cfg = yass.Config.from_yaml(path_to_threshold_config)
    assert cfg.spike_size == 15
