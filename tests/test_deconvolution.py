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


def test_decovnolute_new_pipeline(path_to_config):
    yass.set_config('tests/config_nnet.yaml')
    score, clr_idx, spt = preprocess.run()
    spike_train, spikes_left, templates = process.run(score, clr_idx, spt)
    deconvolute.run(spike_train, spikes_left, templates)
    clean_tmp()
