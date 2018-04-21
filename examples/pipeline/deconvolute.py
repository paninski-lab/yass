import logging

import yass
from yass import preprocess
from yass import detect
from yass import cluster
from yass import templates
from yass import deconvolute

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

# set yass configuration parameters
yass.set_config('config_sample.yaml')

(standarized_path, standarized_params, channel_index,
 whiten_filter) = preprocess.run()

(score, spike_index_clear,
 spike_index_all) = detect.run(standarized_path,
                               standarized_params,
                               channel_index,
                               whiten_filter)


spike_train_clear, tmp_loc, vbParam = cluster.run(
    score, spike_index_clear)

(templates_, spike_train,
 groups, idx_good_templates) = templates.run(
    spike_train_clear, tmp_loc)

spike_train = deconvolute.run(spike_index_all, templates_)
