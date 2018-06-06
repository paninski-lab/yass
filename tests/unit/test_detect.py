from os import path
import pytest
import numpy as np
import yass
from yass import preprocess
from yass import detect
from util import clean_tmp
from util import ReferenceTesting


def test_can_detect_with_threshold(path_to_threshold_config):
    yass.set_config(path_to_threshold_config)
    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    scores, clear, collision = detect.run(standarized_path,
                                          standarized_params,
                                          channel_index,
                                          whiten_filter)
    clean_tmp()


@pytest.mark.xfail
def test_threshold_detector_returns_expected_results(path_to_threshold_config,
                                                     path_to_output_reference):
    np.random.seed(0)

    yass.set_config(path_to_threshold_config)
    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    scores, clear, collision = detect.run(standarized_path,
                                          standarized_params,
                                          channel_index,
                                          whiten_filter)

    path_to_scores = path.join(path_to_output_reference,
                               'detect_threshold_scores.npy')
    path_to_clear = path.join(path_to_output_reference,
                              'detect_threshold_clear.npy')
    path_to_collision = path.join(path_to_output_reference,
                                  'detect_threshold_collision.npy')

    ReferenceTesting.assert_array_almost_equal(scores, path_to_scores,
                                               decimal=4)
    ReferenceTesting.assert_array_equal(clear, path_to_clear)
    ReferenceTesting.assert_array_equal(collision, path_to_collision)

    clean_tmp()


def test_can_detect_with_nnet(path_to_nnet_config):
    yass.set_config(path_to_nnet_config)
    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    scores, clear, collision = detect.run(standarized_path,
                                          standarized_params,
                                          channel_index,
                                          whiten_filter)
    clean_tmp()


def test_nnet_detector_returns_expected_results(path_to_nnet_config,
                                                path_to_output_reference):
    np.random.seed(0)

    yass.set_config(path_to_nnet_config)
    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    scores, clear, collision = detect.run(standarized_path,
                                          standarized_params,
                                          channel_index,
                                          whiten_filter)

    path_to_scores = path.join(path_to_output_reference,
                               'detect_nnet_scores.npy')
    path_to_clear = path.join(path_to_output_reference,
                              'detect_nnet_clear.npy')
    path_to_collision = path.join(path_to_output_reference,
                                  'detect_nnet_collision.npy')

    ReferenceTesting.assert_array_almost_equal(scores, path_to_scores,
                                               decimal=4)
    ReferenceTesting.assert_array_equal(clear, path_to_clear)
    ReferenceTesting.assert_array_equal(collision, path_to_collision)

    clean_tmp()
