"""Script to run pipeline and save results at every step
"""
import numpy as np
import logging
import yass
from yass import preprocess, detect, cluster, templates, deconvolute

np.random.seed(0)

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

# set yass configuration parameters
yass.set_config('config_threshold_49.yaml')

(standarized_path, standarized_params, channel_index,
 whiten_filter) = preprocess.run()

(score, spike_index_clear,
 spike_index_all) = detect.run(standarized_path,
                               standarized_params,
                               channel_index,
                               whiten_filter,
                               save_results=True)


(spike_train_clear,
 tmp_loc, vbParam) = cluster.run(score, spike_index_clear, save_results=True)

(templates_, spike_train,
 groups, idx_good_templates) = templates.run(spike_train_clear, tmp_loc,
                                             save_results=True)

spike_train = deconvolute.run(spike_index_all, templates_)
