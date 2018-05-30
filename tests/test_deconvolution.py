from os import path
import os

import pytest

import yass
from yass import preprocess, detect, cluster, templates, deconvolute
from util import clean_tmp, ReferenceTesting


@pytest.fixture
def path_to_config():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold.yaml')
    return path


def test_decovnolution(path_to_config):
    yass.set_config('tests/config_nnet.yaml')

    (standarized_path,
     standarized_params,
     channel_index,
     whiten_filter) = preprocess.run()

    (score,
     spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        score, spike_index_clear)

    (templates_, spike_train,
     groups, idx_good_templates) = templates.run(
        spike_train_clear, tmp_loc)

    deconvolute.run(spike_index_all, templates_)

    clean_tmp()


def test_deconvolution_returns_expected_results(path_to_config,
                                                path_to_data_folder):
    yass.set_config(path_to_config)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(score, spike_index_clear)

    (templates_, spike_train,
     groups, idx_good_templates) = templates.run(spike_train_clear, tmp_loc)

    spike_train = deconvolute.run(spike_index_all, templates_)

    path_to_spike_train = path.join(path_to_data_folder,
                                    'output_reference',
                                    'spike_train.npy')

    ReferenceTesting.assert_array_equal(spike_train, path_to_spike_train)

    clean_tmp()
