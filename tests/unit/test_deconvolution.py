import util
import yass
from yass import preprocess, cluster, templates, deconvolute


def test_deconvolution(monkeypatch, path_to_nnet_config, make_tmp_folder):
    monkeypatch.setattr('yass.neuralnetwork.KerasModel', util.DummyKerasModel)
    from yass import detect

    yass.set_config(path_to_nnet_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run(output_directory=make_tmp_folder)

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter,
                                   output_directory=make_tmp_folder)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        spike_index_clear,
        output_directory=make_tmp_folder)

    (templates_, spike_train,
     groups, idx_good_templates) = templates.run(
        spike_train_clear, tmp_loc,
        output_directory=make_tmp_folder)

    deconvolute.run(spike_index_all, templates_,
                    output_directory=make_tmp_folder)
