from os import path
from pathlib import Path

from yass.preprocess.filter import _butterworth
from yass.preprocess.standarize import _standard_deviation
from yass.util import load_yaml

import yass
from yass import preprocess

from util import ReferenceTesting


def test_can_apply_butterworth_filter(data):
    _butterworth(data[:, 0], low_frequency=300, high_factor=0.1,
                 order=3, sampling_frequency=20000)


def test_standard_deviation_returns_as_expected(path_to_output_reference,
                                                data):
    sd = _standard_deviation(data, 20000)

    path_to_sd = path.join(path_to_output_reference,
                           'preprocess_sd.npy')

    ReferenceTesting.assert_array_almost_equal(sd, path_to_sd)


def test_filter_does_not_run_if_files_already_exist(path_to_data,
                                                    make_tmp_folder,
                                                    data_info):

    info = data_info['recordings']

    preprocess.butterworth(path_to_data, dtype='int16',
                           n_channels=info['n_channels'],
                           data_order='samples', low_frequency=300,
                           high_factor=0.1, order=3,
                           sampling_frequency=info['sampling_rate'],
                           max_memory='1GB', output_path=make_tmp_folder,
                           output_dtype='float32', processes=1)

    assert preprocess.butterworth.executed

    preprocess.butterworth(path_to_data, dtype='int16',
                           n_channels=info['n_channels'],
                           data_order='samples', low_frequency=300,
                           high_factor=0.1, order=3,
                           sampling_frequency=info['sampling_rate'],
                           max_memory='1GB', output_path=make_tmp_folder,
                           output_dtype='float32', processes=1)

    assert not preprocess.butterworth.executed


def test_standarize_does_not_run_if_files_already_exist(path_to_data,
                                                        data_info,
                                                        make_tmp_folder):

    info = data_info['recordings']

    preprocess.standarize(path_to_data, info['dtype'],
                          info['n_channels'], info['order'],
                          info['sampling_rate'], max_memory='1GB',
                          output_path=make_tmp_folder, output_dtype='float32')

    assert preprocess.standarize.executed

    preprocess.standarize(path_to_data, info['dtype'],
                          info['n_channels'], info['order'],
                          info['sampling_rate'], max_memory='1GB',
                          output_path=make_tmp_folder, output_dtype='float32')

    assert not preprocess.standarize.executed


def test_can_preprocess(path_to_config, make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)
    standarized_path, standarized_params, whiten_filter = preprocess.run()


def test_preprocess_saves_result_in_the_right_folder(path_to_config,
                                                     make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)
    standarized_path, standarized_params, _ = preprocess.run()

    expected = Path(make_tmp_folder, 'preprocess', 'standarized.bin')

    assert str(expected) == standarized_path
    assert expected.is_file()


def test_can_preprocess_in_parallel(path_to_config, make_tmp_folder):
    CONFIG = load_yaml(path_to_config)
    CONFIG['resources']['processes'] = 'max'

    yass.set_config(CONFIG, make_tmp_folder)

    standarized_path, standarized_params, whiten_filter = preprocess.run()


# FIXME: reference testing was deactivated
# def test_preprocess_returns_expected_results(path_to_config,
#                                              path_to_output_reference,
#                                              make_tmp_folder):
#     yass.set_config(path_to_config, make_tmp_folder)
#     standarized_path, standarized_params, whiten_filter = preprocess.run()

#     # load standarized data
#     standarized = np.fromfile(standarized_path,
#                                dtype=standarized_params['dtype'])

#     path_to_standarized = path.join(path_to_output_reference,
#                                      'preprocess_standarized.npy')
#     path_to_whiten_filter = path.join(path_to_output_reference,
#                                       'preprocess_whiten_filter.npy')

#     ReferenceTesting.assert_array_almost_equal(standarized,
#                                                path_to_standarized)
#     ReferenceTesting.assert_array_almost_equal(whiten_filter,
#                                                path_to_whiten_filter)


def test_can_preprocess_without_filtering(path_to_config,
                                          make_tmp_folder):
    CONFIG = load_yaml(path_to_config)
    CONFIG['preprocess'] = dict(apply_filter=False)

    yass.set_config(CONFIG, make_tmp_folder)

    standarized_path, standarized_params, whiten_filter = preprocess.run()
