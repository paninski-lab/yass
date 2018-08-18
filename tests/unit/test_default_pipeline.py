"""
Testing default pipeline
"""
from os import path
import pytest
from yass import pipeline
from util import ReferenceTesting


@pytest.mark.xfail
def test_nnet_pipeline_returns_expected_results(path_to_nnet_config,
                                                path_to_data_folder,
                                                make_tmp_folder):
    spike_train = pipeline.run(path_to_nnet_config, output_dir=make_tmp_folder)

    path_to_reference = path.join(path_to_data_folder,
                                  'output_reference',
                                  'nnet_spike_train.npy')

    ReferenceTesting.assert_array_equal(spike_train, path_to_reference)


@pytest.mark.xfail
def test_threshold_pipeline_returns_expected_results(path_to_threshold_config,
                                                     path_to_data_folder,
                                                     make_tmp_folder):
    spike_train = pipeline.run(path_to_threshold_config,
                               output_dir=make_tmp_folder)

    path_to_reference = path.join(path_to_data_folder,
                                  'output_reference',
                                  'threshold_spike_train.npy')

    ReferenceTesting.assert_array_equal(spike_train, path_to_reference)
