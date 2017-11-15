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
score, clr_idx, spt = preprocess.run()

# run processor
spike_train, spikes_left, templates = process.run(score, clr_idx, spt)

# run deconvolution
spikes = deconvolute.run(spike_train, spikes_left, templates)
