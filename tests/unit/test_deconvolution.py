import os
import yass
from yass import preprocess, cluster, deconvolve, detect


def test_deconvolution(patch_triage_network, path_to_config,
                       make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standardized_path,
     standardized_params) = preprocess.run(
        os.path.join(make_tmp_folder, 'preprocess'))

    spike_index_path = detect.run(
        standardized_path, standardized_params,
        os.path.join(make_tmp_folder, 'detect'))

    fname_templates, fname_spike_train = cluster.run(
        spike_index_path,
        standardized_path,
        standardized_params['dtype'],
        os.path.join(make_tmp_folder, 'cluster'),
        True,
        True)

    (fname_templates,
     fname_spike_train,
     fname_templates_up,
     fname_spike_train_up) = deconvolve.run(
        fname_templates,
        os.path.join(make_tmp_folder,
                     'deconv'),
        standardized_path,
        standardized_params['dtype'])
