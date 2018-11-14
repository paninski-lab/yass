import logging

import yass
from yass import preprocess

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

# set yass configuration parameters
yass.set_config('config_sample.yaml', 'preprocess-example')

# run preprocessor
standardized_path, standardized_params, whiten_filter = preprocess.run()
