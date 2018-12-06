from os import path

import yass
from yass import preprocess
from yass import detect
from yass import cluster
from yass import templates

from util import ReferenceTesting


def test_templates_returns_expected_results(path_to_config,
                                            path_to_output_reference,
                                            make_tmp_folder):

    yass.set_config(path_to_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter)

    (spike_train_clear,
     tmp_loc,
     vbParam) = cluster.run(spike_index_clear)

    (templates_, spike_train,
     groups,
     idx_good_templates) = templates.run(spike_train_clear, tmp_loc,
                                         save_results=True)

    path_to_templates = path.join(path_to_output_reference,
                                  'templates.npy')

    ReferenceTesting.assert_array_equal(templates_, path_to_templates)
