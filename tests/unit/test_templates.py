"""
process.run tests, checking that the pipeline finishes without errors for
several configuration files
"""
import yass
from yass import preprocess
from yass import detect
from yass import cluster
from yass import templates
from yass.detect import threshold


def test_templates(path_to_config, make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   function=threshold.run)

    (spike_train_clear,
     tmp_loc,
     vbParam) = cluster.run(spike_index_clear)

    templates.run(spike_train_clear, tmp_loc)


def test_templates_save_results(path_to_config, make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   function=threshold.run)

    (spike_train_clear,
     tmp_loc,
     vbParam) = cluster.run(spike_index_clear)

    templates.run(spike_train_clear, tmp_loc,
                  save_results=True)


def test_templates_loads_from_disk_if_files_exist(make_tmp_folder,
                                                  path_to_config):
    yass.set_config(path_to_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   function=threshold.run)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        spike_index_clear)

    # save results
    templates.run(spike_train_clear, tmp_loc, save_results=True)

    assert templates.run.executed

    # next time this should not run and just load from files
    templates.run(spike_train_clear, tmp_loc, save_results=True)

    assert not templates.run.executed

# TODO: test template processor
