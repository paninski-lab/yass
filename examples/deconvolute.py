# Note: we are working on a improved version Deconvolution, this old
# pipeline will be removed in the near future. However, migrating to the new
# pipeline requires minimum code changes


import numpy as np

import yass
from yass.preprocessing import Preprocessor
from yass.mainprocess import Mainprocessor
from yass.deconvolute import Deconvolution

cfg = yass.Config.from_yaml('tests/config_nnet.yaml')

pp = Preprocessor(cfg)
score, spike_index_clear, spike_index_collision = pp.process()


mp = Mainprocessor(cfg, score, spike_index_clear, spike_index_collision)
spike_train_clear, spike_index_collision = mp.mainProcess()


dc = Deconvolution(cfg, np.transpose(mp.templates, [1, 0, 2]),
                   spike_index_collision)
spike_train = dc.fullMPMU()

spike_train
