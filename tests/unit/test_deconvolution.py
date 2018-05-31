import pytest
from os import path
import numpy as np
import yass
from yass import preprocess, detect, cluster, templates, deconvolute
from util import clean_tmp, ReferenceTesting


def test_decovnolution(path_to_threshold_config):
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


@pytest.mark.xfail
def test_deconvolution_returns_expected_results(path_to_threshold_config,
                                                path_to_data_folder):
    np.random.seed(0)

    yass.set_config(path_to_threshold_config)

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
