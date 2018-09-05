"""
process.run tests, checking that the pipeline finishes without errors for
several configuration files
"""
# from os import path

# import numpy as np
import pytest

import yass
from yass import preprocess
from yass import detect
from yass import cluster

# from util import ReferenceTesting


def test_cluster(path_to_threshold_config, make_tmp_folder):
    yass.set_config(path_to_threshold_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   output_directory=make_tmp_folder)

    cluster.run(spike_index_clear, output_directory=make_tmp_folder)


# @pytest.mark.xfail
# def test_cluster_returns_expected_results(path_to_threshold_config,
#                                           path_to_data_folder,
#                                           make_tmp_folder):
#     np.random.seed(0)

#     yass.set_config(path_to_threshold_config, make_tmp_folder)

#     (standarized_path,
#      standarized_params,
#      whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

#     (spike_index_clear,
#      spike_index_all) = detect.run(standarized_path,
#                                    standarized_params,
#                                    whiten_filter,
#                                    output_directory=make_tmp_folder)

#     # save results
#     (spike_train, tmp_loc,
#      vbParam) = cluster.run(spike_index_clear,
#                             output_directory=make_tmp_folder)

#     path_to_spike_train = path.join(path_to_data_folder,
#                                     'output_reference',
#                                     'cluster_spike_train.npy')
#     path_to_tmp_loc = path.join(path_to_data_folder,
#                                 'output_reference',
#                                 'cluster_tmp_loc.npy')

#     ReferenceTesting.assert_array_equal(spike_train, path_to_spike_train)
#     ReferenceTesting.assert_array_equal(tmp_loc, path_to_tmp_loc)


def test_cluster_save_results(path_to_threshold_config, make_tmp_folder):

    yass.set_config(path_to_threshold_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   output_directory=make_tmp_folder)

    cluster.run(spike_index_clear, save_results=True,
                output_directory=make_tmp_folder)


def test_cluster_loads_from_disk_if_all_files_exist(caplog,
                                                    path_to_threshold_config,
                                                    make_tmp_folder):

    yass.set_config(path_to_threshold_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   output_directory=make_tmp_folder)

    # save results
    cluster.run(spike_index_clear, save_results=True,
                output_directory=make_tmp_folder)

    assert cluster.run.executed

    # next time this should not run and just load from files
    cluster.run(spike_index_clear, save_results=True,
                output_directory=make_tmp_folder)

    assert not cluster.run.executed


def test_cluster_runs_if_overwrite_is_on(path_to_threshold_config):
    pass


def test_cluster_shows_error_if_empty_config():
    yass.reset_config()

    with pytest.raises(ValueError):
        cluster.run(None, None, None)
