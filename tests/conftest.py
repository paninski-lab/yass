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

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'data/neuropixel.bin')
    d = np.fromfile(path, dtype='int16')
    d = d.reshape(info['observations'], info['n_channels'])
    return d


@pytest.fixture(scope='session')
def path_to_tests():
    path = os.path.dirname(os.path.realpath(__file__))
    return path


@pytest.fixture(scope='session')
def path_to_tmp():
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(path, 'data/tmp/')


@pytest.fixture(scope='session')
def path_to_data():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'data/neuropixel.bin')
    return path


@pytest.fixture(scope='session')
def path_to_geometry():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'data/neuropixel_channels.npy')
    return path


@pytest.fixture(scope='session')
def path_to_examples():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '../examples')
    return path


@pytest.fixture(scope='session')
def path_to_data_folder():
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(path, 'data/')


@pytest.fixture(scope='session')
def path_to_output_reference():
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(path, 'data', 'output_reference')


@pytest.fixture
def path_to_nnet_config(scope='session'):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_nnet.yaml')
    return path


@pytest.fixture
def path_to_threshold_config(scope='session'):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold.yaml')
    return path


@pytest.fixture
def path_to_config_sample(scope='session'):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_sample.yaml')
    return path
