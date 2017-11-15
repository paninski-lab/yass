# Note: we are working on a improved version of the preprocessor, this old
# pipeline will be removed in the near future. However, migrating to the new
# pipeline requires minimum code changes

import yass
from yass.preprocessing import Preprocessor

cfg = yass.Config.from_yaml('tests/config_nnet.yaml')

pp = Preprocessor(cfg)
score, spike_index_clear, spike_index_collision = pp.process()
