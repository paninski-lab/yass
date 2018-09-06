import util
import yass
from yass import preprocess, cluster, templates, deconvolute


def test_deconvolution(monkeypatch, path_to_nnet_config, make_tmp_folder):
    monkeypatch.setattr('yass.neuralnetwork.KerasModel', util.DummyKerasModel)
    from yass import detect

    yass.set_config(path_to_nnet_config, make_tmp_folder)

    (standarized_path,
     standarized_params,
     whiten_filter) = preprocess.run()

    (spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        spike_index_clear)

    (templates_, spike_train,
     groups, idx_good_templates) = templates.run(
        spike_train_clear, tmp_loc)

    deconvolute.run(spike_index_all, templates_)
