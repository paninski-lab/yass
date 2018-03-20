import os
import pytest


@pytest.fixture(scope='session')
def path_to_tests():
    path = os.path.dirname(os.path.realpath(__file__))
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
