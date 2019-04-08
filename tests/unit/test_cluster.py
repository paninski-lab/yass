"""
process.run tests, checking that the pipeline finishes without errors for
some configurations
"""
import os

import yass
from yass import preprocess
from yass import detect
from yass import cluster


def test_cluster_nnet(path_to_config, make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standardized_path,
     standardized_params) = preprocess.run(
        os.path.join(make_tmp_folder, 'preprocess'))

    spike_index_path = detect.run(
        standardized_path, standardized_params,
        os.path.join(make_tmp_folder, 'detect'))

    cluster.run(
        spike_index_path,
        standardized_path,
        standardized_params['dtype'],
        os.path.join(make_tmp_folder, 'cluster'),
        True,
        True)


# def test_cluster_threshold(path_to_config_threshold, make_tmp_folder):
#     yass.set_config(path_to_config_threshold, make_tmp_folder)

#     (standardized_path,
#      standardized_params) = preprocess.run()

#     spike_index_path = detect.run(standardized_path,
#                                   standardized_params)

#     cluster.run(None, spike_index_all)

# def test_cluster_save_results(path_to_config, make_tmp_folder):

#     yass.set_config(path_to_config, make_tmp_folder)

#     (standardized_path,
#      standardized_params,
#      whiten_filter) = preprocess.run()

#     spike_index_all = detect.run(standardized_path,
#                                  standardized_params,
#                                  whiten_filter)

#     cluster.run(None, spike_index_all, save_results=True)


# FIXME: this feature has not been implemented in the new clustering step
# def test_cluster_loads_from_disk_if_all_files_exist(path_to_config,
#                                                     make_tmp_folder):
#     yass.set_config(path_to_config, make_tmp_folder)

#     (standardized_path,
#      standardized_params,
#      whiten_filter) = preprocess.run()

#     spike_index_all = detect.run(standardized_path,
#                                  standardized_params,
#                                  whiten_filter)

#     # save results
#     cluster.run(None, spike_index_all, save_results=True)

#     assert cluster.run.executed

#     # next time this should not run and just load from files
#     cluster.run(None, spike_index_all, save_results=True)

#     assert not cluster.run.executed
