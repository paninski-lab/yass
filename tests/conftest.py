import shutil
import tempfile
import numpy as np
import os
import pytest
from util import PATH_TO_TESTS, seed, dummy_predict_with_threshold


@pytest.fixture(autouse=True)
def setup():
    seed(0)


@pytest.fixture
def patch_triage_network(monkeypatch):
    to_patch = 'yass.neuralnetwork.model.KerasModel.predict_with_threshold'
    monkeypatch.setattr(to_patch, dummy_predict_with_threshold)

    yield


@pytest.fixture()
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


@pytest.fixture()
def data():
    info = data_info()

    path = os.path.join(PATH_TO_TESTS, 'data/retina/data.bin')
    d = np.fromfile(path, dtype='int16')
    d = d.reshape(info['observations'], info['n_channels'])
    return d


@pytest.fixture()
def path_to_tests():
    return PATH_TO_TESTS


@pytest.fixture()
def path_to_performance():
    return os.path.join(PATH_TO_TESTS, 'performance/')


@pytest.fixture
def make_tmp_folder():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture()
def path_to_data():
    return os.path.join(PATH_TO_TESTS, 'data/retina/data.bin')


@pytest.fixture()
def path_to_geometry():
    return os.path.join(PATH_TO_TESTS, 'data/retina/geometry.npy')


@pytest.fixture()
def path_to_data_folder():
    return os.path.join(PATH_TO_TESTS, 'data/')


@pytest.fixture()
def path_to_sample_pipeline_folder():
    return os.path.join(PATH_TO_TESTS, 'data', 'retina',
                        'sample_pipeline_output')


@pytest.fixture()
def path_to_standarized_data():
    return os.path.join(PATH_TO_TESTS, 'data', 'retina',
                        'sample_pipeline_output', 'preprocess',
                        'standarized.bin')


@pytest.fixture()
def path_to_output_reference():
    return os.path.join(PATH_TO_TESTS, 'output_reference')


@pytest.fixture
def path_to_nnet_config():
    return os.path.join(PATH_TO_TESTS, 'config_nnet.yaml')


@pytest.fixture
def path_to_threshold_config():
    return os.path.join(PATH_TO_TESTS, 'config_threshold.yaml')


@pytest.fixture
def path_to_config_sample():
    return os.path.join(PATH_TO_TESTS, 'config_sample.yaml')


@pytest.fixture
def path_to_config_with_wrong_channels():
    return os.path.join(PATH_TO_TESTS, 'config_wrong_channels.yaml')


@pytest.fixture
def path_to_txt_geometry():
    return os.path.join(PATH_TO_TESTS, 'test_files', 'geometry.txt')


@pytest.fixture
def path_to_npy_geometry():
    return os.path.join(PATH_TO_TESTS, 'test_files', 'geometry.npy')
