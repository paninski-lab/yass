import util
from os import path
import yass
from yass import preprocess
from yass import detect
from util import ReferenceTesting


def test_threshold_detector_returns_expected_results(path_to_config_threshold,
                                                     path_to_output_reference,
                                                     make_tmp_folder):
    util.seed(0)

    yass.set_config(path_to_config_threshold, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    clear = detect.run(standarized_path,
                       standarized_params,
                       whiten_filter)

    path_to_clear = path.join(path_to_output_reference,
                              'detect_threshold_clear.npy')

    ReferenceTesting.assert_array_equal(clear, path_to_clear)


def test_nnet_detector_returns_expected_results(path_to_config,
                                                path_to_output_reference,
                                                make_tmp_folder):
    util.seed(0)

    yass.set_config(path_to_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    clear = detect.run(standarized_path,
                       standarized_params,
                       whiten_filter)

    path_to_clear = path.join(path_to_output_reference,
                              'detect_nnet_clear.npy')

    ReferenceTesting.assert_array_equal(clear, path_to_clear)
