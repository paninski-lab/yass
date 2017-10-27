import os.path

import pytest
from yass import geometry


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
