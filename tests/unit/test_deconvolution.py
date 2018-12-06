import yass
from yass import preprocess, cluster, templates, deconvolve, detect


def test_deconvolution(patch_triage_network, path_to_nnet_config,
                       make_tmp_folder):
    yass.set_config(path_to_nnet_config, make_tmp_folder)

    (standardized_path,
     standardized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standardized_path,
                                   standardized_params,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        spike_index_clear)

    (templates_, spike_train,
     groups, idx_good_templates) = templates.run(
        spike_train_clear, tmp_loc)

    deconvolve.run(spike_index_all, templates_)
