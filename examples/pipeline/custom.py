"""
Example for creating a custom YASS pipeline
"""
import logging
import numpy as np

import yass
from yass import preprocess
from yass import detect
from yass import cluster
from yass import templates
from yass import deconvolute

from yass.preprocess.experimental import run as experimental_run

from yass.detect import nnet
from yass.detect import nnet_experimental
from yass.detect import threshold

# just for reproducibility..,
np.random.seed(0)

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

# set yass configuration parameters
yass.set_config('config.yaml', 'custom-example')

# run standarization using the stable implementation (by default)
(standardized_path, standardized_params,
 whiten_filter) = preprocess.run()

# ...or using the experimental code (see source code for details)
(standardized_path, standardized_params,
 whiten_filter) = preprocess.run(function=experimental_run)

# run detection using threshold detector
(spike_index_clear,
 spike_index_all) = detect.run(standardized_path,
                               standardized_params,
                               whiten_filter,
                               function=threshold.run)

# ...or using the neural network detector (see source code for details)
# on changing the network to use
(spike_index_clear,
 spike_index_all) = detect.run(standardized_path,
                               standardized_params,
                               whiten_filter,
                               function=nnet.run)


# ...or using the experimental neural network detector
# (see source code for details) on changing the network to use and the
# difference between this and the stable implementation
(spike_index_clear,
 spike_index_all) = detect.run(standardized_path,
                               standardized_params,
                               whiten_filter,
                               function=nnet_experimental.run)

# the rest is the same, you can customize the pipeline by passing different
# functions
spike_train_clear, tmp_loc, vbParam = cluster.run(spike_index_clear)

(templates_, spike_train,
 groups, idx_good_templates) = templates.run(
    spike_train_clear, tmp_loc)

spike_train = deconvolute.run(spike_index_all, templates_)
