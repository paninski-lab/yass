"""
process.run tests, checking that the pipeline finishes without errors for
several configuration files
"""
from os import path

import pytest
import numpy as np

import yass
from yass import preprocess
from yass import detect
from yass import cluster
from yass import templates
from yass import reset_config

from util import clean_tmp, ReferenceTesting


def teardown_function(function):
    reset_config()
    clean_tmp()


def test_templates(path_to_threshold_config):
    yass.set_config(path_to_threshold_config)

    standarized_path, standarized_params, whiten_filter = preprocess.run()

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        score, spike_index_clear)

    templates.run(spike_train_clear, tmp_loc)

    clean_tmp()


def test_templates_save_results(path_to_threshold_config):

    yass.set_config(path_to_threshold_config)

    standarized_path, standarized_params, whiten_filter = preprocess.run()

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        score, spike_index_clear)

    templates.run(spike_train_clear, tmp_loc, save_results=True)

    clean_tmp()


def test_templates_loads_from_disk_if_files_exist(caplog,
                                                  path_to_threshold_config):

    yass.set_config(path_to_threshold_config)

    standarized_path, standarized_params, whiten_filter = preprocess.run()

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        score, spike_index_clear)

    # save results
    templates.run(spike_train_clear, tmp_loc, save_results=True)

    assert templates.run.executed

    # next time this should not run and just load from files
    templates.run(spike_train_clear, tmp_loc, save_results=True)

    assert not templates.run.executed


@pytest.mark.xfail
def test_templates_returns_expected_results(path_to_threshold_config,
                                            path_to_data_folder):
    np.random.seed(0)

    yass.set_config(path_to_threshold_config)

    standarized_path, standarized_params, whiten_filter = preprocess.run()

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(score, spike_index_clear)

    (templates_, spike_train,
     groups, idx_good_templates) = templates.run(spike_train_clear, tmp_loc)

    path_to_templates = path.join(path_to_data_folder,
                                  'output_reference',
                                  'templates.npy')

    ReferenceTesting.assert_array_equal(templates_, path_to_templates)

    clean_tmp()


def test_new_process_shows_error_if_empty_config():
    with pytest.raises(ValueError):
        cluster.run(None, None)
