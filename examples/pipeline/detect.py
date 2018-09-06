"""
Detecting spikes
"""

import logging

import yass
from yass import preprocess
from yass import detect

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

# set yass configuration parameters
yass.set_config('config_sample.yaml', 'detect-example')

# run preprocessor
standarized_path, standarized_params, whiten_filter = preprocess.run()

# run detection
clear, collision = detect.run(standarized_path,
                              standarized_params,
                              whiten_filter)
