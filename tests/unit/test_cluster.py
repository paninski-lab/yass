"""
process.run tests, checking that the pipeline finishes without errors for
some configurations
"""
import yass
from yass import preprocess
from yass import detect
from yass import cluster


def test_cluster_threshold(path_to_threshold_config, make_tmp_folder):
    yass.set_config(path_to_threshold_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter)

    cluster.run(spike_index_clear)


def test_cluster_nnet(patch_triage_network, path_to_nnet_config,
                      make_tmp_folder):
    yass.set_config(path_to_nnet_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter)

    cluster.run(spike_index_clear)


def test_cluster_save_results(path_to_threshold_config, make_tmp_folder):

    yass.set_config(path_to_threshold_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter)

    cluster.run(spike_index_clear, save_results=True)


def test_cluster_loads_from_disk_if_all_files_exist(path_to_threshold_config,
                                                    make_tmp_folder):
    yass.set_config(path_to_threshold_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter)

    # save results
    cluster.run(spike_index_clear, save_results=True)

    assert cluster.run.executed

    # next time this should not run and just load from files
    cluster.run(spike_index_clear, save_results=True)

    assert not cluster.run.executed
