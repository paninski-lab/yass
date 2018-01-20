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
spike_train_clear, templates, spike_index_collision = process.run(score,
                                                                  clr_idx, spt)

# run deconvolution
spike_train = deconvolute.run(spike_train_clear, templates,
                              spike_index_collision)
