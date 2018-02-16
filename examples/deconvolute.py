import logging

import yass
from yass import preprocess
from yass import process
from yass import deconvolute

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

# set yass configuration parameters
yass.set_config('tests/config_nnet.yaml')

# run preprocessor
score, spike_index_clear, spike_index_all = preprocess.run()

# run processor
spike_train_clear, templates = process.run(score, spike_index_clear)

# run deconvolution
spike_train = deconvolute.run(spike_index_all, templates)
