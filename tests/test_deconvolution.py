import os

import numpy as np
import pytest

import yass
from yass.preprocessing import Preprocessor
from yass.mainprocess import Mainprocessor
from yass.deconvolute import Deconvolution_depreciated

from yass import preprocess
from yass import process
from yass import deconvolute


@pytest.fixture
def path_to_config():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'config_threshold.yaml')
    return path


def test_deconvolute(path_to_config):
    cfg = yass.Config.from_yaml(path_to_config)

    pp = Preprocessor(cfg)
    score, clr_idx, spt = pp.process()

    mp = Mainprocessor(cfg, score, clr_idx, spt)
    spike_train, spt_left = mp.mainProcess()

    dc = Deconvolution_depreciated(cfg, np.transpose(mp.templates, [1, 0, 2]), spt_left)
    dc.fullMPMU()


def test_decovnolute_new_pipeline(path_to_config):
    yass.set_config('tests/config_nnet.yaml')
    score, clr_idx, spt = preprocess.run()
    spike_train, spikes_left, templates = process.run(score, clr_idx, spt)
    deconvolute.run(spike_train, spikes_left, templates)
