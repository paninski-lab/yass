"""
Test speed for pipeline steps
"""
import yass
from yass import preprocess, detect, cluster, templates
from util import clean_tmp


def test_templates_speed(path_to_config):
    yass.set_config(path_to_config)

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    spike_train_clear, tmp_loc, vbParam = cluster.run(
        score, spike_index_clear)

    templates.run(spike_train_clear, tmp_loc)

    clean_tmp()
