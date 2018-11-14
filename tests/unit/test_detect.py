import yass
from yass import preprocess
from yass import detect


def test_can_detect_with_threshold(path_to_threshold_config, make_tmp_folder):
    yass.set_config(path_to_threshold_config, make_tmp_folder)

    (standardized_path,
     standardized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standardized_path,
                                   standardized_params,
                                   whiten_filter)


def test_can_detect_with_nnet(path_to_nnet_config, make_tmp_folder):
    yass.set_config(path_to_nnet_config, make_tmp_folder)

    (standardized_path,
     standardized_params,
     whiten_filter) = preprocess.run()

    detect.run(standardized_path,
               standardized_params,
               whiten_filter)
