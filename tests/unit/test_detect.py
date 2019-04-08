import os

import yass
from yass import preprocess
from yass import detect


# def test_can_detect_with_threshold(path_to_config_threshold,
#                                    make_tmp_folder):
#     yass.set_config(path_to_config_threshold, make_tmp_folder)

#     (standardized_path,
#      standardized_params,
#      whiten_filter) = preprocess.run()

#     detect.run(standardized_path,
#                standardized_params,
#                whiten_filter)


def test_can_detect_with_nnet(path_to_config, make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standardized_path,
     standardized_params) = preprocess.run(
        os.path.join(make_tmp_folder, 'preprocess'))

    detect.run(
        standardized_path, standardized_params,
        os.path.join(make_tmp_folder, 'detect'))

# this is causing segmentation fault error when ran locally (travis runs ok)
# def test_can_detect_with_nnet_experimental(path_to_config, make_tmp_folder):
#     yass.set_config(path_to_config, make_tmp_folder)

#     (standardized_path,
#      standardized_params,
#      whiten_filter) = preprocess.run()

#     detect.run(standardized_path,
#                standardized_params,
#                whiten_filter,
#                function=nnet_experimental.run)
