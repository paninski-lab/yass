import numpy as np
from os import path

import yass
from yass import preprocess, cluster, deconvolve, detect, read_config


def test_deconvolution(patch_triage_network, path_to_config,
                       make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standardized_path,
     standardized_params,
     whiten_filter) = preprocess.run()

    spike_index_all = detect.run(standardized_path,
                                 standardized_params,
                                 whiten_filter)

    cluster.run(None, spike_index_all)

    CONFIG = read_config()
    TMP_FOLDER = CONFIG.path_to_output_directory

    path_to_spike_train_cluster = path.join(TMP_FOLDER,
                                            'spike_train_cluster.npy')
    spike_train_cluster = np.load(path_to_spike_train_cluster)
    templates_cluster = np.load(path.join(TMP_FOLDER, 'templates_cluster.npy'))

    spike_train, postdeconv_templates = deconvolve.run(spike_train_cluster,
                                                       templates_cluster)
