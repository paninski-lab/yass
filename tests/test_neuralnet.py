"""
neuralnetwork module tests
"""
import os.path as path

import numpy as np
import yaml

import yass
from yass.batch import RecordingsReader, BatchProcessor
from yass import neuralnetwork
from yass.geometry import make_channel_index


def test_can_use_neural_network_detector(path_to_tests):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    data = RecordingsReader(path.join(path_to_tests,
                                      'data/standarized.bin'),
                            loader='array').data.T

    channel_index = make_channel_index(CONFIG.neigh_channels,
                                       CONFIG.geom)

    whiten_filter = np.tile(np.eye(channel_index.shape[1], dtype='float32')[
        np.newaxis, :, :], [channel_index.shape[0], 1, 1])

    detection_th = CONFIG.detect.neural_network_detector.threshold_spike
    triage_th = CONFIG.detect.neural_network_triage.threshold_collision
    detection_fname = CONFIG.detect.neural_network_detector.filename
    ae_fname = CONFIG.detect.neural_network_autoencoder.filename
    triage_fname = CONFIG.detect.neural_network_triage.filename
    (x_tf, output_tf, NND,
     NNAE, NNT) = neuralnetwork.prepare_nn(channel_index,
                                           whiten_filter,
                                           detection_th,
                                           triage_th,
                                           detection_fname,
                                           ae_fname,
                                           triage_fname
                                           )

    neuralnetwork.run_detect_triage_featurize(data, x_tf, output_tf,
                                              NND, NNAE, NNT)


def test_splitting_in_batches_does_not_affect_result(path_to_tests):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    PATH_TO_DATA = path.join(path_to_tests, 'data/standarized.bin')

    data = RecordingsReader(PATH_TO_DATA, loader='array').data

    with open(path.join(path_to_tests, 'data/standarized.yaml')) as f:
        PARAMS = yaml.load(f)

    channel_index = make_channel_index(CONFIG.neigh_channels,
                                       CONFIG.geom)

    whiten_filter = np.tile(np.eye(channel_index.shape[1], dtype='float32')[
        np.newaxis, :, :], [channel_index.shape[0], 1, 1])

    detection_th = CONFIG.detect.neural_network_detector.threshold_spike
    triage_th = CONFIG.detect.neural_network_triage.threshold_collision
    detection_fname = CONFIG.detect.neural_network_detector.filename
    ae_fname = CONFIG.detect.neural_network_autoencoder.filename
    triage_fname = CONFIG.detect.neural_network_triage.filename
    (x_tf, output_tf, NND,
     NNAE, NNT) = neuralnetwork.prepare_nn(channel_index,
                                           whiten_filter,
                                           detection_th,
                                           triage_th,
                                           detection_fname,
                                           ae_fname,
                                           triage_fname
                                           )

    # buffer size makes sure we can detect spikes if they appear at the end of
    # any batch
    bp = BatchProcessor(PATH_TO_DATA, PARAMS['dtype'], PARAMS['n_channels'],
                        PARAMS['data_format'], '100KB',
                        buffer_size=CONFIG.spike_size)
    mc = bp.multi_channel_apply
    res = mc(
        neuralnetwork.run_detect_triage_featurize,
        mode='memory',
        cleanup_function=neuralnetwork.fix_indexes,
        x_tf=x_tf,
        output_tf=output_tf,
        NND=NND,
        NNAE=NNAE,
        NNT=NNT)

    scores_batch = np.concatenate([element[0] for element in res], axis=0)
    clear_batch = np.concatenate([element[1] for element in res], axis=0)
    collision_batch = np.concatenate([element[2] for element in res], axis=0)

    (scores, clear,
     collision) = neuralnetwork.run_detect_triage_featurize(data, x_tf,
                                                            output_tf,
                                                            NND, NNAE,
                                                            NNT)

    np.testing.assert_array_equal(clear_batch, clear)
    np.testing.assert_array_equal(collision_batch, collision)
    np.testing.assert_array_equal(scores_batch, scores)


def test_can_train_nnet(path_to_tests):
    pass
