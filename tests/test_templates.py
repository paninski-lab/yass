"""
process.run tests, checking that the pipeline finishes without errors for
several configuration files
"""

import os

import pytest

import yass
from yass import preprocess
from yass import detect
from yass import cluster
from yass import templates
from yass import reset_config

from util import clean_tmp


def teardown_function(function):
    reset_config()
    clean_tmp()


@pytest.fixture
def path_to_config():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold.yaml')
    return path


def test_templates(path_to_config):
    yass.set_config(path_to_config)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        score, spike_index_clear)

    templates.run(spike_train_clear, tmp_loc)

    clean_tmp()


def test_templates_save_results(path_to_config):

    yass.set_config(path_to_config)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(spike_index_clear)

    templates.run(spike_train_clear, tmp_loc, save_results=True)

    clean_tmp()


def test_templates_loads_from_disk_if_all_files_exist(caplog, path_to_config):

    yass.set_config(path_to_config)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(spike_index_clear)

    # save results
    templates.run(spike_train_clear, tmp_loc, save_results=True)

    assert templates.run.executed

    # next time this should not run and just load from files
    templates.run(spike_train_clear, tmp_loc, save_results=True)

    assert not templates.run.executed


def test_new_process_shows_error_if_empty_config():
    with pytest.raises(ValueError):
        cluster.run(None, None)
