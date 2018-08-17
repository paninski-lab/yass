import numpy as np
import os
import pytest
from util import make_tmp, clean_tmp, PATH_TO_TESTS


@pytest.fixture(scope='session')
def data_info():
    d = dict()

    d['spike_size_ms'] = 1
    d['srate'] = 30000
    d['spike_size'] = int(np.round(d['spike_size_ms']*d['srate']/2000))
    d['BUFF'] = d['spike_size'] * 2
    d['scale_to_save'] = 100
    d['n_features'] = 3
    d['n_channels'] = 10
    d['observations'] = 10000
    d['data_order'] = 'samples'
    d['dtype'] = 'int16'
    d['sampling_frequency'] = 20000

    return d


@pytest.fixture(scope='session')
def data():
    info = data_info()

    path = os.path.join(PATH_TO_TESTS, 'data/neuropixel.bin')
    d = np.fromfile(path, dtype='int16')
    d = d.reshape(info['observations'], info['n_channels'])
    return d


@pytest.fixture(scope='session')
def path_to_tests():
    return PATH_TO_TESTS


@pytest.fixture(scope='session')
def path_to_performance():
    return os.path.join(PATH_TO_TESTS, 'performance/')


@pytest.fixture(scope='session')
def path_to_tmp():
    # FIXME: remove this and replace for tmp_folder, that one creates and
    # deletes the tmp/ folder automatically
    return os.path.join(PATH_TO_TESTS, 'data/tmp/')


@pytest.fixture(scope='session')
def tmp_folder():
    make_tmp()

    yield os.path.join(PATH_TO_TESTS, 'data/tmp/')

    clean_tmp()


@pytest.fixture(scope='session')
def path_to_data():
    return os.path.join(PATH_TO_TESTS, 'data/neuropixel/data.bin')


@pytest.fixture(scope='session')
def path_to_geometry():
    return os.path.join(PATH_TO_TESTS, 'data/neuropixel/geometry.npy')


@pytest.fixture(scope='session')
def path_to_examples():
    return os.path.join(PATH_TO_TESTS, '../examples')


@pytest.fixture(scope='session')
def path_to_data_folder():
    return os.path.join(PATH_TO_TESTS, 'data/')


@pytest.fixture(scope='session')
def path_to_sample_pipeline_folder():
    return os.path.join(PATH_TO_TESTS, 'data', 'sample_pipeline_output')


@pytest.fixture(scope='session')
def path_to_standarized_data():
    return os.path.join(PATH_TO_TESTS, 'data', 'sample_pipeline_output',
                        'preprocess', 'standarized.bin')


@pytest.fixture(scope='session')
def path_to_output_reference():
    return os.path.join(PATH_TO_TESTS, 'data', 'output_reference')


@pytest.fixture
def path_to_nnet_config(scope='session'):
    return os.path.join(PATH_TO_TESTS, 'config_nnet.yaml')


@pytest.fixture
def path_to_threshold_config(scope='session'):
    return os.path.join(PATH_TO_TESTS, 'config_threshold.yaml')


@pytest.fixture
def path_to_config_sample(scope='session'):
    return os.path.join(PATH_TO_TESTS, 'config_sample.yaml')


@pytest.fixture
def path_to_config_with_wrong_channels(scope='session'):
    return os.path.join(PATH_TO_TESTS, 'config_wrong_channels.yaml')


@pytest.fixture
def path_to_txt_geometry(scope='session'):
    return os.path.join(PATH_TO_TESTS, 'data/geometry.txt')


@pytest.fixture
def path_to_npy_geometry(scope='session'):
    return os.path.join(PATH_TO_TESTS, 'data/neuropixel/geometry.npy')
