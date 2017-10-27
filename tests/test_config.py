import os

import pytest

import yass


@pytest.fixture
def path_to_config():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold.yaml')
    return path


def test_can_init(path_to_config):
    yass.Config.from_yaml(path_to_config)


def test_cannot_modify_once_initialized(path_to_config):
    cfg = yass.Config.from_yaml(path_to_config)

    with pytest.raises(AttributeError):
        cfg.param = 1


def test_extra_parameters_are_computed(path_to_config):
    cfg = yass.Config.from_yaml(path_to_config)
    assert cfg.spikeSize == 15
