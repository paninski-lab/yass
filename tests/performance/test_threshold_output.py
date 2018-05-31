from os import path
import numpy as np
import yass
from yass import preprocess, detect, cluster, templates, deconvolute

PATHT_TO_REF = '/Users/Edu/data/yass/ref49/'


def test_threshold_output(path_to_tests):
    """Test that pipeline using threshold detector returns the same results
    """
    yass.set_config(path.join(path_to_tests, 'config_threshold_49.yaml'))

    (standarized_path, standarized_params, channel_index,
     whiten_filter) = preprocess.run()

    path_to_standarized = path.join(PATHT_TO_REF,
                                    'preprocess', 'standarized.bin')
    path_to_channel_index = path.join(PATHT_TO_REF,
                                      'preprocess', 'channel_index.npy')
    path_to_whitening = path.join(PATHT_TO_REF, 'preprocess', 'whitening.npy')

    channel_index_saved = np.load(path_to_channel_index)
    whitening_saved = np.load(path_to_whitening)

    # test preprocess output
    np.testing.assert_array_equal(channel_index_saved, channel_index)
    np.testing.assert_array_equal(whitening_saved, whiten_filter)
    # TODO: check standarized data

    (score, spike_index_clear,
     spike_index_all) = detect.run(standarized_path,
                                   standarized_params,
                                   channel_index,
                                   whiten_filter)

    path_to_scores = path.join(PATHT_TO_REF, 'detect', 'scores_clear.npy')
    path_to_spike_index_clear = path.join(PATHT_TO_REF, 'detect',
                                          'spike_index_clear.npy')
    path_to_spike_index_all = path.join(PATHT_TO_REF, 'detect',
                                        'spike_index_all.npy')

    scores_saved = np.load(path_to_scores)
    spike_index_clear_saved = np.load(path_to_spike_index_clear)
    spike_index_all_saved = np.load(path_to_spike_index_all)

    # test detect output
    np.testing.assert_array_equal(scores_saved, score)
    np.testing.assert_array_equal(spike_index_clear_saved, spike_index_clear)
    np.testing.assert_array_equal(spike_index_all_saved, spike_index_all)
    # TODO: add missing

    (spike_train_clear,
     tmp_loc, vbParam) = cluster.run(score, spike_index_clear)

    path_to_spike_train_cluster = path.join(PATHT_TO_REF, 'cluster',
                                            'spike_train_cluster.npy')
    spike_train_cluster_saved = np.load(path_to_spike_train_cluster)

    # test cluster
    np.testing.assert_array_equal(spike_train_cluster_saved, spike_train_clear)

    # test templates
    (templates_, spike_train,
     groups, idx_good_templates) = templates.run(spike_train_clear, tmp_loc,
                                                 save_results=True)

    path_to_templates = path.join(PATHT_TO_REF, 'templates', 'templates.npy')
    templates_saved = np.load(path_to_templates)

    np.testing.assert_array_equal(templates_saved, templates_)

    # test deconvolution
    spike_train = deconvolute.run(spike_index_all, templates_)

    path_to_spike_train = path.join(PATHT_TO_REF, 'spike_train.npy')
    spike_train_saved = np.load(path_to_spike_train)

    np.testing.assert_array_equal(spike_train_saved, spike_train)
