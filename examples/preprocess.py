import logging

import yass
from yass import preprocess

# configure logging module to get useful information
logging.basicConfig(level=logging.DEBUG)

# set yass configuration parameters
yass.set_config('tests/config_nnet.yaml')

# run preprocessor
score, spike_index_clear, spike_index_collision = preprocess.run()
