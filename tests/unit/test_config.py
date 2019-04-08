import pytest
import yass


def test_can_initialize_config(path_to_config):
    yass.Config.from_yaml(path_to_config)


def test_throws_error_if_channels_mismatch(path_to_config_with_wrong_channels):
    with pytest.raises(ValueError):
        yass.Config.from_yaml(path_to_config_with_wrong_channels)


def test_cannot_modify_once_initialized(path_to_config):
    cfg = yass.Config.from_yaml(path_to_config)

    with pytest.raises(AttributeError):
        cfg.param = 1
