import pytest
import numpy as np
from yass import geometry
from yass.geometry import (parse, find_channel_neighbors,
                           n_steps_neigh_channels)
from yass.threshold import detect


def test_can_load_npy(path_to_npy_geometry):
    geom = geometry.parse(path_to_npy_geometry, n_channels=10)
    assert geom.shape == (10, 2)


def test_can_load_txt(path_to_txt_geometry):
    geom = geometry.parse(path_to_txt_geometry, n_channels=7)
    assert geom.shape == (7, 2)


def test_raise_error_if_unsupported_ext():
    with pytest.raises(ValueError):
        geometry.parse('unsupported.mat', n_channels=7)


def test_raises_error_if_npy_with_wrong_channels(path_to_npy_geometry):
    with pytest.raises(ValueError):
        geometry.parse(path_to_npy_geometry, n_channels=500)


def test_raises_error_if_txt_with_wrong_channels(path_to_txt_geometry):
    with pytest.raises(ValueError):
        geometry.parse(path_to_txt_geometry, n_channels=500)


def test_can_parse(data_info, path_to_geometry):
    parse(path_to_geometry, data_info['recordings']['n_channels'])


def test_can_compute_channel_neighbors(data_info, path_to_geometry):
    geometry = parse(path_to_geometry, data_info['recordings']['n_channels'])
    find_channel_neighbors(geometry, radius=70)


def test_can_compute_n_steps_neighbors(data_info, path_to_geometry):
    geometry = parse(path_to_geometry, data_info['recordings']['n_channels'])
    neighbors = find_channel_neighbors(geometry, radius=70)
    n_steps_neigh_channels(neighbors, steps=2)


def test_can_use_threshold_detector(data, data_info, path_to_geometry):
    geometry = parse(path_to_geometry, data_info['recordings']['n_channels'])
    neighbors = find_channel_neighbors(geometry, radius=70)

    # FIXME: using the same formula from yass/config/config.py, might be
    # better to avoid having this hardcoded
    spike_size = int(np.round(data_info['recordings']['spike_size_ms'] *
                     data_info['recordings']['sampling_rate'] / (2 * 1000)))

    detect._threshold(data, neighbors, spike_size, 5)
