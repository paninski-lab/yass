from os import path
import pytest
import numpy as np
import yass
from yass import preprocess
from yass import detect
from util import ReferenceTesting


def test_can_detect_with_threshold(path_to_threshold_config, make_tmp_folder):
    yass.set_config(path_to_threshold_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   output_directory=make_tmp_folder)


@pytest.mark.xfail
def test_threshold_detector_returns_expected_results(path_to_threshold_config,
                                                     path_to_output_reference,
                                                     make_tmp_folder):
    np.random.seed(0)

    yass.set_config(path_to_threshold_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    clear, collision = detect.run(standarized_path,
                                  standarized_params,
                                  whiten_filter,
                                  output_directory=make_tmp_folder)

    path_to_clear = path.join(path_to_output_reference,
                              'detect_threshold_clear.npy')
    path_to_collision = path.join(path_to_output_reference,
                                  'detect_threshold_collision.npy')

    ReferenceTesting.assert_array_equal(clear, path_to_clear)
    ReferenceTesting.assert_array_equal(collision, path_to_collision)


def test_can_detect_with_nnet(path_to_nnet_config, make_tmp_folder):
    yass.set_config(path_to_nnet_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    detect.run(standarized_path,
               standarized_params,
               whiten_filter,
               output_directory=make_tmp_folder)


@pytest.mark.xfail
def test_nnet_detector_returns_expected_results(path_to_nnet_config,
                                                path_to_output_reference,
                                                make_tmp_folder):
    np.random.seed(0)

    yass.set_config(path_to_nnet_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    clear, collision = detect.run(standarized_path,
                                  standarized_params,
                                  whiten_filter,
                                  output_directory=make_tmp_folder)

    path_to_clear = path.join(path_to_output_reference,
                              'detect_nnet_clear.npy')
    path_to_collision = path.join(path_to_output_reference,
                                  'detect_nnet_collision.npy')

    ReferenceTesting.assert_array_equal(clear, path_to_clear)
    ReferenceTesting.assert_array_equal(collision, path_to_collision)
