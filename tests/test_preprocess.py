from os import path
import os

import numpy as np
import pytest


from yass.preprocess.filter import _butterworth
from yass.preprocess import whiten

# FIXME: MOVE THIS TO A DIFFERENT TEST SUITE
from yass.geometry import (parse, find_channel_neighbors,
                           n_steps_neigh_channels,
                           make_channel_index)

from yass.threshold import detect
from yass.preprocess.standarize import _standard_deviation
from yass.util import load_yaml

import yass
from yass import preprocess

from util import clean_tmp
from util import ReferenceTesting


spike_sizeMS = 1
srate = 30000
spike_size = int(np.round(spike_sizeMS*srate/2000))
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


def teardown_function(function):
    clean_tmp()


def test_can_apply_butterworth_filter(data):
    _butterworth(data[:, 0], low_frequency=300, high_factor=0.1,
                 order=3, sampling_frequency=20000)


def test_standard_deviation_returns_as_espected(path_to_output_reference,
                                                data):
    sd = _standard_deviation(data, 20000)

    path_to_sd = path.join(path_to_output_reference,
                           'preprocess_sd.npy')

    ReferenceTesting.assert_array_almost_equal(sd, path_to_sd)


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
    detect._threshold(data, neighbors, spike_size, 5)


def test_can_compute_whiten_matrix(data, path_to_geometry):
    geometry = parse(path_to_geometry, n_channels)
    neighbors = find_channel_neighbors(geometry, radius=70)
    channel_index = make_channel_index(neighbors, geometry)

    whiten._matrix(data, channel_index, spike_size)


def test_can_preprocess(path_to_threshold_config):
    yass.set_config(path_to_threshold_config)
    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()


def test_can_preprocess_in_parallel(path_to_threshold_config):
    CONFIG = load_yaml(path_to_threshold_config)
    CONFIG['resources']['processes'] = 'max'

    yass.set_config(CONFIG)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()


def test_preprocess_returns_expected_results(path_to_threshold_config,
                                             path_to_output_reference):
    yass.set_config(path_to_threshold_config)
    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    # load standarized data
    standarized = np.fromfile(standarized_path,
                              dtype=standarized_params['dtype'])

    path_to_standarized = path.join(path_to_output_reference,
                                    'preprocess_standarized.npy')
    path_to_whiten_filter = path.join(path_to_output_reference,
                                      'preprocess_whiten_filter.npy')
    path_to_channel_index = path.join(path_to_output_reference,
                                      'preprocess_channel_index.npy')

    ReferenceTesting.assert_array_equal(standarized, path_to_standarized)
    ReferenceTesting.assert_array_equal(whiten_filter, path_to_whiten_filter)
    ReferenceTesting.assert_array_equal(channel_index, path_to_channel_index)

    clean_tmp()


def test_can_preprocess_without_filtering(path_to_threshold_config):
    CONFIG = load_yaml(path_to_threshold_config)
    CONFIG['preprocess']['apply_filter'] = False

    yass.set_config(CONFIG)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()
