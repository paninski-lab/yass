"""
Tests for functions that create training data for the neural networks
"""
import os.path as path

import pytest
import numpy as np

import yass
from yass.templates.util import get_templates
from yass.augment.noise import noise_cov
from yass.augment.util import (make_from_templates, make_collided, make_noise,
                               make_spatially_misaligned,
                               make_temporally_misaligned)
from yass.batch import RecordingsReader


@pytest.fixture()
def templates_uncropped(path_to_config, make_tmp_folder,
                        path_to_sample_pipeline_folder,
                        path_to_standarized_data):
    spike_train = np.array([100, 0,
                            150, 0,
                            200, 1,
                            250, 1,
                            300, 2,
                            350, 2]).reshape(-1, 2)

    yass.set_config(path_to_config, make_tmp_folder)
    CONFIG = yass.read_config()

    spike_train = np.load(path.join(path_to_sample_pipeline_folder,
                                    'spike_train.npy'))

    n_spikes, _ = spike_train.shape

    weighted_spike_train = np.hstack((spike_train,
                                      np.ones((n_spikes, 1), 'int32')))

    templates_uncropped, _ = get_templates(weighted_spike_train,
                                           path_to_standarized_data,
                                           CONFIG.resources.max_memory,
                                           4*CONFIG.spike_size)

    templates_uncropped_ = np.transpose(templates_uncropped, (2, 1, 0))

    return templates_uncropped_


def test_can_make_clean(templates_uncropped):
    make_from_templates(templates_uncropped,
                        min_amplitude=2,
                        max_amplitude=10,
                        n_per_template=100)


def test_can_make_collided(templates_uncropped):
    x_clean = make_from_templates(templates_uncropped, min_amplitude=2,
                                  max_amplitude=10,
                                  n_per_template=100)

    make_collided(x_clean, x_clean, n_per_spike=1,
                  max_shift=3,
                  min_shift=2)


def test_can_make_spatially_misaligned(templates_uncropped):
    x_clean = make_from_templates(templates_uncropped, min_amplitude=2,
                                  max_amplitude=10,
                                  n_per_template=100)

    make_spatially_misaligned(x_clean, n_per_spike=1)


def test_can_make_temporally_misaligned(templates_uncropped):
    x_clean = make_from_templates(templates_uncropped,
                                  min_amplitude=2,
                                  max_amplitude=10,
                                  n_per_template=100)

    make_temporally_misaligned(x_clean, n_per_spike=1)


def test_can_compute_noise_cov(path_to_tests, path_to_standarized_data):
    recordings = RecordingsReader(path_to_standarized_data,
                                  loader='array')._data

    spatial_SIG, temporal_SIG = noise_cov(recordings,
                                          temporal_size=10,
                                          sample_size=100,
                                          threshold=3.0,
                                          window_size=10)


def test_can_make_noise(path_to_tests, path_to_standarized_data):
    recordings = RecordingsReader(path_to_standarized_data,
                                  loader='array')._data

    spatial_SIG, temporal_SIG = noise_cov(recordings,
                                          temporal_size=10,
                                          sample_size=100,
                                          threshold=3.0,
                                          window_size=10)

    make_noise(10, spatial_SIG, temporal_SIG)
