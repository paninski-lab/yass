from os import path
import numpy as np
import yass
from yass import preprocess

PATH_TO_TMP = '/Users/Edu/data/yass/tmp/'


def test_preprocess_output(path_to_tests):
    path_to_standarized = path.join(PATH_TO_TMP, 'standarized.bin')
    path_to_channel_index = path.join(PATH_TO_TMP, 'channel_index.npy')
    path_to_whitening = path.join(PATH_TO_TMP, 'whitening.npy')

    yass.set_config(path.join(path_to_tests, 'config_threshold_49.yaml'))

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    channel_index_saved = np.load(path_to_channel_index)
    whitening_saved = np.load(path_to_whitening)

    np.testing.assert_array_equal(channel_index_saved, channel_index)
    np.testing.assert_array_equal(whitening_saved, whiten_filter)
