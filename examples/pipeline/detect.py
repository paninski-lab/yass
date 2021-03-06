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
yass.set_config('config.yaml', 'example-detect')

# run preprocessor
standardized_path, standardized_params, whiten_filter = preprocess.run()

# run detection
clear, collision = detect.run(standardized_path,
                              standardized_params,
                              whiten_filter,
                              if_file_exists='overwrite')
