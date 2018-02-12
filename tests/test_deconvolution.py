import os

import pytest

import yass

from yass import preprocess
from yass import process
from yass import deconvolute

from util import clean_tmp


@pytest.fixture
def path_to_config():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold.yaml')
    return path


def test_decovnolution(path_to_config):
    yass.set_config('tests/config_nnet.yaml')
    clear_scores, spike_index_clear, spike_index_all = preprocess.run()
    (spike_train_clear,
     templates) = process.run(clear_scores, spike_index_clear)
    deconvolute.run(spike_index_all, templates)
    clean_tmp()
