import pytest
import yass


def test_can_initialize_threshold_config(path_to_threshold_config):
    yass.Config.from_yaml(path_to_threshold_config)


def test_can_initialize_nnet_config(path_to_nnet_config):
    yass.Config.from_yaml(path_to_nnet_config)


def test_can_initialize_config_sample(path_to_threshold_config):
    yass.Config.from_yaml(path_to_threshold_config)


def test_throws_error_if_channels_mismatch(path_to_config_with_wrong_channels):
    with pytest.raises(ValueError):
        yass.Config.from_yaml(path_to_config_with_wrong_channels)


def test_cannot_modify_once_initialized(path_to_threshold_config):
    cfg = yass.Config.from_yaml(path_to_threshold_config)

    with pytest.raises(AttributeError):
        cfg.param = 1


def test_extra_parameters_are_computed(path_to_threshold_config):
    cfg = yass.Config.from_yaml(path_to_threshold_config)
    assert cfg.spike_size == 15
