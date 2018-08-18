"""
Tests for functions that create training data for the neural networks
"""
import os.path as path
import random

import numpy as np

import yass
from yass.augment import make
from yass.templates.util import get_templates
from yass.templates.crop import crop_and_align_templates
from yass.augment.noise import noise_cov
from yass.augment.util import (make_from_templates, make_collided,
                               make_misaligned, make_noise)


def test_can_make_training_data(path_to_tests, path_to_sample_pipeline_folder):

    np.random.seed(0)
    random.seed(0)

    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    spike_train = np.load(path.join(path_to_sample_pipeline_folder,
                                    'spike_train.npy'))
    chosen_templates = np.unique(spike_train[:, 1])
    min_amplitude = 4
    max_amplitude = 60
    n_spikes_to_make = 10

    templates = make.load_templates(path_to_sample_pipeline_folder,
                                    spike_train, CONFIG, chosen_templates)

    path_to_standarized = path.join(path_to_sample_pipeline_folder,
                                    'preprocess', 'standarized.bin')

    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make.training_data(CONFIG, templates,
                                      min_amplitude, max_amplitude,
                                      n_spikes_to_make,
                                      path_to_standarized)


def test_can_make_clean(path_to_tests, path_to_standarized_data,
                        path_to_sample_pipeline_folder):

    np.random.seed(0)
    random.seed(0)

    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
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

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    make_from_templates(templates_uncropped, min_amplitude=2, max_amplitude=10,
                        n_per_template=100)


def test_can_make_collided(path_to_tests, path_to_standarized_data,
                           path_to_sample_pipeline_folder):

    np.random.seed(0)
    random.seed(0)

    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
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

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    x_clean = make_from_templates(templates_uncropped, min_amplitude=2,
                                  max_amplitude=10,
                                  n_per_template=100)

    make_collided(x_clean, n_per_spike=1,
                  multi_channel=True,
                  max_shift=CONFIG.spike_size)


def test_can_make_misaligned(path_to_tests, path_to_standarized_data,
                             path_to_sample_pipeline_folder):

    np.random.seed(0)
    random.seed(0)

    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
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

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    x_clean = make_from_templates(templates_uncropped, min_amplitude=2,
                                  max_amplitude=10, n_per_template=100)

    make_misaligned(x_clean,
                    misalign_ratio=1,
                    misalign_ratio2=1,
                    max_shift=2 * CONFIG.spike_size,
                    multi_channel=True)


def test_can_compute_noise_cov(path_to_tests, path_to_standarized_data,
                               path_to_sample_pipeline_folder):

    np.random.seed(0)
    random.seed(0)

    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
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

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    noise_cov(path_to_standarized_data,
              CONFIG.neigh_channels,
              CONFIG.geom,
              templates_uncropped.shape[1])


def test_can_make_noise(path_to_tests, path_to_standarized_data,
                        path_to_sample_pipeline_folder):

    np.random.seed(0)
    random.seed(0)

    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
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

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    templates = crop_and_align_templates(templates_uncropped,
                                         CONFIG.spike_size,
                                         CONFIG.neigh_channels,
                                         CONFIG.geom)

    spatial_SIG, temporal_SIG = noise_cov(path_to_standarized_data,
                                          CONFIG.neigh_channels,
                                          CONFIG.geom,
                                          templates.shape[1])

    x_clean = make_from_templates(templates, min_amplitude=2, max_amplitude=10,
                                  n_per_template=100)

    make_noise(x_clean.shape, spatial_SIG, temporal_SIG)
