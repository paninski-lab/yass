import yass
from yass import preprocess
from yass import detect


def test_can_detect_with_threshold(path_to_config_threshold, make_tmp_folder):
    yass.set_config(path_to_config_threshold, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    detect.run(standarized_path,
               standarized_params,
               whiten_filter)


def test_can_detect_with_nnet(path_to_config, make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    detect.run(standarized_path,
               standarized_params,
               whiten_filter)

# this is causing segmentation fault error when ran locally (travis runs ok)
# def test_can_detect_with_nnet_experimental(path_to_config, make_tmp_folder):
#     yass.set_config(path_to_config, make_tmp_folder)

#     (standarized_path,
#      standarized_params,
#      whiten_filter) = preprocess.run()

#     detect.run(standarized_path,
#                standarized_params,
#                whiten_filter,
#                function=nnet_experimental.run)
