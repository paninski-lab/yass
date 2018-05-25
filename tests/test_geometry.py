import os.path
import pytest
from yass import geometry
from yass.geometry import (parse, find_channel_neighbors,
                           n_steps_neigh_channels,
                           make_channel_index)
from yass.preprocess import whiten
from yass.threshold import detect

here = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def path_to_txt():
    path = os.path.join(here, 'data/geometry.txt')
    return path


@pytest.fixture
def path_to_npy():
    path = os.path.join(here, 'data/geometry.npy')
    return path


def test_can_load_npy(path_to_npy):
    geom = geometry.parse(path_to_npy, n_channels=374)
    assert geom.shape == (374, 2)


def test_can_load_txt(path_to_txt):
    geom = geometry.parse(path_to_txt, n_channels=7)
    assert geom.shape == (7, 2)


def test_raise_error_if_unsupported_ext():
    with pytest.raises(ValueError):
        geometry.parse('unsupported.mat', n_channels=7)


def test_raises_error_if_npy_with_wrong_channels(path_to_npy):
    with pytest.raises(ValueError):
        geometry.parse(path_to_npy, n_channels=500)


def test_raises_error_if_txt_with_wrong_channels(path_to_txt):
    with pytest.raises(ValueError):
        geometry.parse(path_to_txt, n_channels=500)


def test_can_parse(data_info, path_to_geometry):
    parse(path_to_geometry, data_info['n_channels'])


def test_can_compute_channel_neighbors(data_info, path_to_geometry):
    geometry = parse(path_to_geometry, data_info['n_channels'])
    find_channel_neighbors(geometry, radius=70)


def test_can_compute_n_steps_neighbors(data_info, path_to_geometry):
    geometry = parse(path_to_geometry, data_info['n_channels'])
    neighbors = find_channel_neighbors(geometry, radius=70)
    n_steps_neigh_channels(neighbors, steps=2)


def test_can_use_threshold_detector(data, data_info, path_to_geometry):
    geometry = parse(path_to_geometry, data_info['n_channels'])
    neighbors = find_channel_neighbors(geometry, radius=70)
    detect._threshold(data, neighbors, data_info['spike_size'], 5)


def test_can_compute_whiten_matrix(data, data_info, path_to_geometry):
    geometry = parse(path_to_geometry, data_info['n_channels'])
    neighbors = find_channel_neighbors(geometry, radius=70)
    channel_index = make_channel_index(neighbors, geometry)

    whiten._matrix(data, channel_index, data_info['spike_size'])
