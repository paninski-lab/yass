import logging

import yass
from yass import preprocess
from yass import process

# configure logging module to get useful information
logging.basicConfig(level=logging.DEBUG)

# set yass configuration parameters
yass.set_config('config_sample.yaml')

# run preprocessor
score, spike_index_clear, spike_index_all = preprocess.run()

# run processor
spike_train_clear, templates = process.run(score, clear_spike_index)
