import os

import pytest

import yass
from yass import preprocess, detect, cluster, templates, deconvolute
from util import clean_tmp


@pytest.fixture
def path_to_config():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold.yaml')
    return path


def test_decovnolution(path_to_config):
    yass.set_config('tests/config_nnet.yaml')

    (standarized_path,
     standarized_params,
     channel_index,
     whiten_filter) = preprocess.run()

    (score,
     spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    spike_train_clear = cluster.run(score, spike_index_clear)

    templates_ = templates.run(spike_train_clear)

    deconvolute.run(spike_index_all, templates_)

    clean_tmp()
