"""
process.run tests, checking that the pipeline finishes without errors for
several configuration files
"""
from os import path
import os

import pytest

import yass
from yass import preprocess
from yass import detect
from yass import cluster
from yass import reset_config

from util import clean_tmp
from util import ReferenceTesting


def teardown_function(function):
    reset_config()
    clean_tmp()


@pytest.fixture
def path_to_config():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold.yaml')
    return path


def test_cluster(path_to_config):
    yass.set_config(path_to_config)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    cluster.run(score, spike_index_clear)

    clean_tmp()


def test_cluster_returns_expected_results(path_to_config,
                                          path_to_data_folder):
    yass.set_config(path_to_config)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    spike_train, tmp_loc, vbParam = cluster.run(score, spike_index_clear)

    path_to_spike_train = path.join(path_to_data_folder,
                                    'output_reference',
                                    'cluster_spike_train.npy')
    path_to_tmp_loc = path.join(path_to_data_folder,
                                'output_reference',
                                'cluster_tmp_loc.npy')

    ReferenceTesting.assert_array_equal(spike_train, path_to_spike_train)
    ReferenceTesting.assert_array_equal(tmp_loc, path_to_tmp_loc)

    clean_tmp()


def test_cluster_save_results(path_to_config):

    yass.set_config(path_to_config)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    cluster.run(score, spike_index_clear, save_results=True)

    clean_tmp()


def test_cluster_loads_from_disk_if_all_files_exist(caplog, path_to_config):

    yass.set_config(path_to_config)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    # save results
    cluster.run(score, spike_index_clear, save_results=True)

    assert cluster.run.executed

    # next time this should not run and just load from files
    cluster.run(score, spike_index_clear, save_results=True)

    assert not cluster.run.executed


def test_cluster_runs_if_overwrite_is_on(path_to_config):
    pass


def test_new_process_shows_error_if_empty_config():
    with pytest.raises(ValueError):
        cluster.run(None, None)
