import logging

import yass
from yass import preprocess

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

# set yass configuration parameters
yass.set_config('config_sample.yaml')

# run preprocessor
(standarized_path, standarized_params, channel_index,
 whiten_filter) = preprocess.run(if_file_exists='skip')
