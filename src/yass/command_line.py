import os
import argparse

import numpy as np
import logging

from . import set_config
from . import preprocess
from . import process
from . import deconvolute
from . import read_config

def main():
    # configure logging module to get useful information
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Run YASS.')
    parser.add_argument('config', type=str,
                        help='Path to configuration file')

    args = parser.parse_args()

    # set yass configuration parameters
    set_config(args.config)
    CONFIG = read_config()

    # run preprocessor
    score, spike_index_clear, spike_index_collision = preprocess.run()

    # run processor
    spike_train_clear, templates, spike_index_collision = process.run(score,
        spike_index_clear, spike_index_collision)

    # run deconvolution
    spike_train = deconvolute.run(spike_train_clear, templates,
        spike_index_collision)

    path_to_file = os.path.join(CONFIG.root, CONFIG.spikeTrainName)
    np.savetxt(path_to_file, spike_train, fmt='%i, %i')
    logger.info('Done, spike train saved in: {}'.format(path_to_file))
