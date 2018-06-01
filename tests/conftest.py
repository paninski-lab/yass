import numpy as np
import os
import pytest


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

    path = os.path.join(path_to_tests(), 'data/neuropixel.bin')
    d = np.fromfile(path, dtype='int16')
    d = d.reshape(info['observations'], info['n_channels'])
    return d


@pytest.fixture(scope='session')
def path_to_tests():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope='session')
def path_to_performance():
    return os.path.join(path_to_tests(), 'performance/')


@pytest.fixture(scope='session')
def path_to_tmp():
    return os.path.join(path_to_tests(), 'data/tmp/')


@pytest.fixture(scope='session')
def path_to_data():
    return os.path.join(path_to_tests(), 'data/neuropixel.bin')


@pytest.fixture(scope='session')
def path_to_geometry():
    return os.path.join(path_to_tests(), 'data/neuropixel_channels.npy')


@pytest.fixture(scope='session')
def path_to_examples():
    return os.path.join(path_to_tests(), '../examples')


@pytest.fixture(scope='session')
def path_to_data_folder():
    return os.path.join(path_to_tests(), 'data/')


@pytest.fixture(scope='session')
def path_to_output_reference():
    return os.path.join(path_to_tests(), 'data', 'output_reference')


@pytest.fixture
def path_to_nnet_config(scope='session'):
    return os.path.join(path_to_tests(), 'config_nnet.yaml')


@pytest.fixture
def path_to_threshold_config(scope='session'):
    return os.path.join(path_to_tests(), 'config_threshold.yaml')


@pytest.fixture
def path_to_config_sample(scope='session'):
    return os.path.join(path_to_tests(), 'config_sample.yaml')


@pytest.fixture
def path_to_txt_geometry(scope='session'):
    return os.path.join(path_to_tests(), 'data/geometry.txt')


@pytest.fixture
def path_to_npy_geometry(scope='session'):
    return os.path.join(path_to_tests(), 'data/geometry.npy')
