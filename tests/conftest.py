import shutil
import tempfile
import numpy as np
import os
import pytest
from util import PATH_TO_TESTS


@pytest.fixture(scope='session')
def data_info():
    d = dict()

    d['spike_size_ms'] = 1.5
    d['sampling_frequency'] = 20000
    d['spike_size'] = int(np.round(d['spike_size_ms'] *
                                   d['sampling_frequency']/2000))
    d['BUFF'] = d['spike_size'] * 2
    d['n_features'] = 3
    d['n_channels'] = 49
    d['observations'] = 20000
    d['data_order'] = 'samples'
    d['dtype'] = 'int16'

    return d


@pytest.fixture(scope='session')
def data():
    info = data_info()

    path = os.path.join(PATH_TO_TESTS, 'data/retina/data.bin')
    d = np.fromfile(path, dtype='int16')
    d = d.reshape(info['observations'], info['n_channels'])
    return d


@pytest.fixture(scope='session')
def path_to_tests():
    return PATH_TO_TESTS


@pytest.fixture(scope='session')
def path_to_performance():
    return os.path.join(PATH_TO_TESTS, 'performance/')


@pytest.fixture
def make_tmp_folder():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture(scope='session')
def path_to_data():
    return os.path.join(PATH_TO_TESTS, 'data/retina/data.bin')


@pytest.fixture(scope='session')
def path_to_standarized():
    return os.path.join(PATH_TO_TESTS, 'data/standarized.bin')


@pytest.fixture(scope='session')
def path_to_geometry():
    return os.path.join(PATH_TO_TESTS, 'data/retina/geometry.npy')


@pytest.fixture(scope='session')
def path_to_data_folder():
    return os.path.join(PATH_TO_TESTS, 'data/')


@pytest.fixture(scope='session')
def path_to_sample_pipeline_folder():
    return os.path.join(PATH_TO_TESTS, 'data', 'retina',
                        'sample_pipeline_output')


@pytest.fixture(scope='session')
def path_to_standarized_data():
    return os.path.join(PATH_TO_TESTS, 'data', 'retina',
                        'sample_pipeline_output', 'preprocess',
                        'standarized.bin')


@pytest.fixture(scope='session')
def path_to_output_reference():
    return os.path.join(PATH_TO_TESTS, 'output_reference')


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
    return os.path.join(PATH_TO_TESTS, 'test_files', 'geometry.txt')


@pytest.fixture
def path_to_npy_geometry(scope='session'):
    return os.path.join(PATH_TO_TESTS, 'test_files', 'geometry.npy')
