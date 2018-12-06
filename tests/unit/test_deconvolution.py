import yass
from yass import preprocess, cluster, templates, deconvolute, detect
from yass.detect import nnet


def test_deconvolution(patch_triage_network, path_to_config,
                       make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   function=nnet.run)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        spike_index_clear)

    (templates_, spike_train,
     groups, idx_good_templates) = templates.run(
        spike_train_clear, tmp_loc)

    deconvolute.run(spike_index_all, templates_)
