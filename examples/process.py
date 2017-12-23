import logging

import yass
from yass import preprocess
from yass import process

# configure logging module to get useful information
logging.basicConfig(level=logging.DEBUG)

# set yass configuration parameters
yass.set_config('tests/config_sample.yaml')

# run preprocessor
score, clr_idx, spt = preprocess.run()

# run processor
(spike_train_clear, templates,
 spike_index_collision) = process.run(score, clr_idx, spt)
