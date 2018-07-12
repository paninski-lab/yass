"""
Tests for functions that create training data for the neural networks
"""
import os.path as path

import numpy as np

import yass
from yass.augment import make
from yass.templates.util import get_templates
from yass.templates.crop import crop_and_align_templates
from yass.augment.noise import noise_cov
from yass.augment.util import (make_clean, make_collided,
                               make_misaligned, make_noise)


spike_train = np.array([100, 0,
                        150, 0,
                        200, 1,
                        250, 1,
                        300, 2,
                        350, 2]).reshape(-1, 2)

chosen_templates = [0, 1, 2]
min_amplitude = 2
max_amplitude = 10
n_spikes_to_make = 500

filters = [8, 4]


def test_can_make_training_data(path_to_tests, path_to_sample_pipeline_folder):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    make.training_data(CONFIG, spike_train, chosen_templates,
                       min_amplitude, max_amplitude, n_spikes_to_make,
                       data_folder=path_to_sample_pipeline_folder)


# FIXME: move this test to test_templates
def test_can_crop_and_align_templates(path_to_tests, path_to_standarized_data):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    n_spikes, _ = spike_train.shape

    weighted_spike_train = np.hstack((spike_train,
                                      np.ones((n_spikes, 1), 'int32')))

    templates_uncropped, _ = get_templates(weighted_spike_train,
                                           path_to_standarized_data,
                                           CONFIG.resources.max_memory,
                                           4*CONFIG.spike_size)

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    crop_and_align_templates(templates_uncropped,
                             CONFIG.spike_size,
                             CONFIG.neigh_channels, CONFIG.geom)


def test_can_make_clean(path_to_tests, path_to_standarized_data):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    n_spikes, _ = spike_train.shape

    weighted_spike_train = np.hstack((spike_train,
                                      np.ones((n_spikes, 1), 'int32')))

    templates_uncropped, _ = get_templates(weighted_spike_train,
                                           path_to_standarized_data,
                                           CONFIG.resources.max_memory,
                                           4*CONFIG.spike_size)

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    make_clean(templates_uncropped, min_amplitude=2, max_amplitude=10, nk=100)


def test_can_make_collided(path_to_tests, path_to_standarized_data):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    n_spikes, _ = spike_train.shape

    weighted_spike_train = np.hstack((spike_train,
                                      np.ones((n_spikes, 1), 'int32')))

    templates_uncropped, _ = get_templates(weighted_spike_train,
                                           path_to_standarized_data,
                                           CONFIG.resources.max_memory,
                                           4*CONFIG.spike_size)

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    x_clean = make_clean(templates_uncropped, min_amplitude=2,
                         max_amplitude=10,
                         nk=100)

    make_collided(x_clean, collision_ratio=1,
                  multi_channel=True,
                  max_shift=CONFIG.spike_size)


def test_can_make_misaligned(path_to_tests, path_to_standarized_data):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    n_spikes, _ = spike_train.shape

    weighted_spike_train = np.hstack((spike_train,
                                      np.ones((n_spikes, 1), 'int32')))

    templates_uncropped, _ = get_templates(weighted_spike_train,
                                           path_to_standarized_data,
                                           CONFIG.resources.max_memory,
                                           4*CONFIG.spike_size)

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    x_clean = make_clean(templates_uncropped, min_amplitude=2,
                         max_amplitude=10, nk=100)

    make_misaligned(x_clean,
                    templates_uncropped,
                    max_shift=2 * CONFIG.spike_size,
                    misalign_ratio=1,
                    misalign_ratio2=1,
                    multi=True,
                    nneigh=templates_uncropped.shape[2])


def test_can_compute_noise_cov(path_to_tests, path_to_standarized_data):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    n_spikes, _ = spike_train.shape

    weighted_spike_train = np.hstack((spike_train,
                                      np.ones((n_spikes, 1), 'int32')))

    templates_uncropped, _ = get_templates(weighted_spike_train,
                                           path_to_standarized_data,
                                           CONFIG.resources.max_memory,
                                           4*CONFIG.spike_size)

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    noise_cov(path_to_standarized_data,
              CONFIG.neigh_channels,
              CONFIG.geom,
              templates_uncropped.shape[1])


def test_can_make_noise(path_to_tests, path_to_standarized_data):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    n_spikes, _ = spike_train.shape

    weighted_spike_train = np.hstack((spike_train,
                                      np.ones((n_spikes, 1), 'int32')))

    templates_uncropped, _ = get_templates(weighted_spike_train,
                                           path_to_standarized_data,
                                           CONFIG.resources.max_memory,
                                           4*CONFIG.spike_size)

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    templates = crop_and_align_templates(templates_uncropped,
                                         CONFIG.spike_size,
                                         CONFIG.neigh_channels,
                                         CONFIG.geom)

    spatial_SIG, temporal_SIG = noise_cov(path_to_standarized_data,
                                          CONFIG.neigh_channels,
                                          CONFIG.geom,
                                          templates.shape[1])

    x_clean = make_clean(templates, min_amplitude=2, max_amplitude=10, nk=100)

    make_noise(x_clean, noise_ratio=10, templates=templates,
               spatial_SIG=spatial_SIG, temporal_SIG=temporal_SIG)
