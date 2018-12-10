import logging

import yass
from yass import preprocess

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

# set yass configuration parameters
yass.set_config('config.yaml', 'example-preprocess')

# run preprocessor
(standarized_path,
 standarized_params,
 whiten_filter) = preprocess.run()
