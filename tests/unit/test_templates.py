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

from util import ReferenceTesting


def test_templates(path_to_threshold_config, make_tmp_folder):
    yass.set_config(path_to_threshold_config)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   output_directory=make_tmp_folder)

    (spike_train_clear,
     tmp_loc,
     vbParam) = cluster.run(spike_index_clear,
                            output_directory=make_tmp_folder)

    templates.run(spike_train_clear, tmp_loc,
                  output_directory=make_tmp_folder)


def test_templates_save_results(path_to_threshold_config, make_tmp_folder):
    yass.set_config(path_to_threshold_config)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   output_directory=make_tmp_folder)

    (spike_train_clear,
     tmp_loc,
     vbParam) = cluster.run(spike_index_clear,
                            output_directory=make_tmp_folder)

    templates.run(spike_train_clear, tmp_loc,
                  output_directory=make_tmp_folder,
                  save_results=True)


def test_templates_loads_from_disk_if_files_exist(make_tmp_folder,
                                                  path_to_threshold_config):
    yass.set_config(path_to_threshold_config)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   output_directory=make_tmp_folder)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        spike_index_clear, output_directory=make_tmp_folder)

    # save results
    templates.run(spike_train_clear, tmp_loc, save_results=True,
                  output_directory=make_tmp_folder)

    assert templates.run.executed

    # next time this should not run and just load from files
    templates.run(spike_train_clear, tmp_loc, save_results=True,
                  output_directory=make_tmp_folder)

    assert not templates.run.executed


@pytest.mark.xfail
def test_templates_returns_expected_results(path_to_threshold_config,
                                            path_to_output_reference,
                                            make_tmp_folder):
    np.random.seed(0)

    yass.set_config(path_to_threshold_config)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   output_directory=make_tmp_folder)

    (spike_train_clear,
     tmp_loc,
     vbParam) = cluster.run(spike_index_clear,
                            output_directory=make_tmp_folder)

    (templates_, spike_train,
     groups,
     idx_good_templates) = templates.run(spike_train_clear, tmp_loc,
                                         output_directory=make_tmp_folder,
                                         save_results=True)

    path_to_templates = path.join(path_to_output_reference,
                                  'templates.npy')

    ReferenceTesting.assert_array_equal(templates_, path_to_templates)


# TODO: test template processor
