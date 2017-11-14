import os

import numpy as np
import pytest


from yass.preprocess.filter import butterworth, whitening_matrix, whitening

# FIXME: MOVE THIS TO A DIFFERENT TEST SUITE
from yass.geometry import (parse, find_channel_neighbors,
                           n_steps_neigh_channels)

from yass.preprocess.detect import threshold_detection
from yass.preprocess.waveform import get_waveforms
from yass.preprocess.standarize import standarize

import yass
from yass import preprocess
from yass.preprocessing import Preprocessor

spikeSizeMS = 1
srate = 30000
spike_size = int(np.round(spikeSizeMS*srate/2000))
BUFF = spike_size * 2
scale_to_save = 100
n_features = 3
n_channels = 10
observations = 10000


@pytest.fixture
def data():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'data/neuropixel.bin')
    d = np.fromfile(path, dtype='int16')
    d = d.reshape(observations, n_channels)
    return d


@pytest.fixture
def path_to_geometry():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'data/neuropixel_channels.npy')
    return path


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


@pytest.fixture
def path_to_nn_config():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_nnet.yaml')
    return path


def test_can_apply_butterworth_filter(data):
    butterworth(data, low_freq=300, high_factor=0.1,
                order=3, sampling_freq=20000)


def test_can_standarize(data):
    standarize(data, srate)


def test_can_parse(path_to_geometry):
    parse(path_to_geometry, n_channels)


def test_can_compute_channel_neighbors(path_to_geometry):
    geometry = parse(path_to_geometry, n_channels)
    find_channel_neighbors(geometry, radius=70)


def test_can_compute_n_steps_neighbors(path_to_geometry):
    geometry = parse(path_to_geometry, n_channels)
    neighbors = find_channel_neighbors(geometry, radius=70)
    n_steps_neigh_channels(neighbors, steps=2)


def test_can_use_threshold_detector(data, path_to_geometry):
    geometry = parse(path_to_geometry, n_channels)
    neighbors = find_channel_neighbors(geometry, radius=70)
    threshold_detection(data, neighbors, spike_size, 5)


def test_can_whiten_data(data, path_to_geometry):
    geometry = parse(path_to_geometry, n_channels)
    neighbors = find_channel_neighbors(geometry, radius=70)
    q = whitening_matrix(data, neighbors, spike_size)
    whitening(data, q)


def test_can_preprocess_data(path_to_config):
    cfg = yass.Config.from_yaml(path_to_config)
    pp = Preprocessor(cfg)
    score, clr_idx, spt = pp.process()


def test_can_preprocess_data_1k(path_to_config_1k):
    cfg = yass.Config.from_yaml(path_to_config_1k)
    pp = Preprocessor(cfg)
    score, clr_idx, spt = pp.process()


def test_can_preprocess_data_with_nnet(path_to_nn_config):
    cfg = yass.Config.from_yaml(path_to_nn_config)
    pp = Preprocessor(cfg)
    score, clr_idx, spt = pp.process()


def test_can_preprocess(path_to_config):
    yass.set_config(path_to_config)
    score, spike_index_clear, spike_index_collision = preprocess.run()


def test_can_preprocess_1k(path_to_config_1k):
    yass.set_config(path_to_config_1k)
    score, spike_index_clear, spike_index_collision = preprocess.run()


def test_can_preprocess_with_nnet(path_to_nn_config):
    yass.set_config(path_to_nn_config)
    score, spike_index_clear, spike_index_collision = preprocess.run()
