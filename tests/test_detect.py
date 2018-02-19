import yass

from yass import preprocess
from yass import detect
from util import clean_tmp


def test_can_detect_with_threshold(path_to_threshold_config):
    yass.set_config(path_to_threshold_config)
    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    scores, clear, collision = detect.run(standarized_path,
                                          standarized_params,
                                          channel_index,
                                          whiten_filter)
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
