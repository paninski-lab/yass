import yass
from yass import preprocess
from yass import detect
from yass.detect import nnet, threshold, nnet_experimental


def test_can_detect_with_threshold(path_to_config, make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   function=threshold.run)


def test_can_detect_with_nnet(path_to_config, make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    detect.run(standarized_path,
               standarized_params,
               whiten_filter,
               function=nnet.run)


def test_can_detect_with_nnet_experimental(path_to_config, make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    detect.run(standarized_path,
               standarized_params,
               whiten_filter,
               function=nnet_experimental.run)
