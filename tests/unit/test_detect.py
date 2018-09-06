import yass
from yass import preprocess
from yass import detect


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


def test_can_detect_with_nnet(path_to_nnet_config, make_tmp_folder):
    yass.set_config(path_to_nnet_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    detect.run(standarized_path,
               standarized_params,
               whiten_filter,
               output_directory=make_tmp_folder)
