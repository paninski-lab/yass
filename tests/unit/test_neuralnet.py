"""
neuralnetwork module tests
"""
import os.path as path

import numpy as np
import tensorflow as tf
import yaml

import yass
from yass.batch import RecordingsReader, BatchProcessor
from yass import neuralnetwork
from yass.neuralnetwork import NeuralNetDetector, NeuralNetTriage, AutoEncoder
from yass.geometry import make_channel_index, n_steps_neigh_channels
from yass.augment import make_training_data
from yass.explore import RecordingExplorer


spike_train = np.array([100, 0,
                        150, 0,
                        200, 1,
                        250, 1,
                        300, 2,
                        350, 2]).reshape(-1, 2)

chosen_templates = [0, 1, 2]
min_amplitude = 2
n_spikes = 500

filters = [8, 4]


def test_can_train_detector(path_to_tests, path_to_sample_pipeline_folder,
                            tmp_folder):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amplitude, n_spikes,
                                      path_to_sample_pipeline_folder)

    _, waveform_length, n_neighbors = x_detect.shape

    path_to_model = path.join(tmp_folder, 'detect-net.ckpt')

    detector = NeuralNetDetector(path_to_model, filters,
                                 waveform_length, n_neighbors,
                                 threshold=0.5,
                                 channel_index=CONFIG.channel_index,
                                 n_iter=10)

    detector.fit(x_detect, y_detect)


def test_can_train_triage(path_to_tests, path_to_sample_pipeline_folder,
                          tmp_folder):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amplitude, n_spikes,
                                      path_to_sample_pipeline_folder)

    _, waveform_length, n_neighbors = x_triage.shape

    path_to_model = path.join(tmp_folder, 'triage-net.ckpt')

    triage = NeuralNetTriage(path_to_model, filters,
                             waveform_length, n_neighbors,
                             threshold=0.5,
                             n_iter=10)

    triage.fit(x_detect, y_detect)


def test_can_train_autoencoder(path_to_tests, path_to_sample_pipeline_folder,
                               tmp_folder):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amplitude, n_spikes,
                                      path_to_sample_pipeline_folder)

    _, waveform_length = x_ae.shape

    path_to_model = path.join(tmp_folder, 'ae.ckpt')

    autoencoder = AutoEncoder(path_to_model, waveform_length, n_features=3)
    autoencoder.fit(x_ae)


def test_can_reload_detector(path_to_tests, path_to_sample_pipeline_folder,
                             tmp_folder):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amplitude, n_spikes,
                                      path_to_sample_pipeline_folder)

    _, waveform_length, n_neighbors = x_detect.shape

    path_to_model = path.join(tmp_folder, 'detect-net.ckpt')

    detector = NeuralNetDetector(path_to_model, filters,
                                 waveform_length, n_neighbors,
                                 threshold=0.5,
                                 channel_index=CONFIG.channel_index,
                                 n_iter=10)

    detector.fit(x_detect, y_detect)

    NeuralNetDetector.load(path_to_model, threshold=0.5,
                           channel_index=CONFIG.channel_index)


def test_can_reload_triage(path_to_tests, path_to_sample_pipeline_folder,
                           tmp_folder):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amplitude, n_spikes,
                                      path_to_sample_pipeline_folder)

    _, waveform_length, n_neighbors = x_triage.shape

    path_to_model = path.join(tmp_folder, 'triage-net.ckpt')

    triage = NeuralNetTriage(path_to_model, filters,
                             waveform_length, n_neighbors,
                             threshold=0.5,
                             n_iter=10)

    triage.fit(x_detect, y_detect)

    NeuralNetTriage.load(path_to_model, threshold=0.5)


def test_can_reload_autoencoder(path_to_tests, path_to_sample_pipeline_folder,
                                tmp_folder):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amplitude, n_spikes,
                                      path_to_sample_pipeline_folder)

    _, waveform_length = x_ae.shape

    path_to_model = path.join(tmp_folder, 'ae.ckpt')

    autoencoder = AutoEncoder(path_to_model, waveform_length, n_features=3)
    autoencoder.fit(x_ae)

    AutoEncoder.load(path_to_model)


def test_can_use_detector_triage_ae_after_fit(path_to_tests,
                                              path_to_sample_pipeline_folder,
                                              tmp_folder,
                                              path_to_standarized_data):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amplitude, n_spikes,
                                      path_to_sample_pipeline_folder)

    _, waveform_length, n_neighbors = x_detect.shape

    path_to_model = path.join(tmp_folder, 'detect-net.ckpt')
    detector = NeuralNetDetector(path_to_model, filters,
                                 waveform_length, n_neighbors,
                                 threshold=0.5,
                                 channel_index=CONFIG.channel_index,
                                 n_iter=10)
    detector.fit(x_detect, y_detect)

    path_to_model = path.join(tmp_folder, 'triage-net.ckpt')
    triage = NeuralNetTriage(path_to_model, filters,
                             waveform_length, n_neighbors,
                             threshold=0.5,
                             n_iter=10)
    triage.fit(x_detect, y_detect)

    path_to_model = path.join(tmp_folder, 'ae.ckpt')
    autoencoder = AutoEncoder(path_to_model, waveform_length, n_neighbors,
                              n_features=3)
    autoencoder.fit(x_ae)

    data = RecordingExplorer(path_to_standarized_data).reader.data

    output_names = ('spike_index', 'waveform', 'probability')

    (spike_index, waveform,
        proba) = detector.predict_recording(data, output_names=output_names)

    detector.predict(x_detect)

    triage.predict(waveform[:, :, :n_neighbors])

    autoencoder.predict(x_ae)


def test_can_use_detect_triage_ae_after_reload(path_to_tests,
                                               path_to_sample_pipeline_folder,
                                               tmp_folder,
                                               path_to_standarized_data):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    (x_detect, y_detect,
     x_triage, y_triage,
     x_ae, y_ae) = make_training_data(CONFIG, spike_train, chosen_templates,
                                      min_amplitude, n_spikes,
                                      path_to_sample_pipeline_folder)

    _, waveform_length, n_neighbors = x_detect.shape

    path_to_model = path.join(tmp_folder, 'detect-net.ckpt')

    detector = NeuralNetDetector(path_to_model, filters,
                                 waveform_length, n_neighbors,
                                 threshold=0.5,
                                 channel_index=CONFIG.channel_index,
                                 n_iter=10)
    detector.fit(x_detect, y_detect)
    detector = NeuralNetDetector.load(path_to_model, threshold=0.5,
                                      channel_index=CONFIG.channel_index)

    triage = NeuralNetTriage(path_to_model, filters,
                             waveform_length, n_neighbors,
                             threshold=0.5,
                             n_iter=10)

    triage.fit(x_detect, y_detect)
    triage = NeuralNetTriage.load(path_to_model, threshold=0.5)

    autoencoder = AutoEncoder(path_to_model, waveform_length, n_neighbors,
                              n_features=3)
    autoencoder.fit(x_ae)
    autoencoder = AutoEncoder.load(path_to_model)

    data = RecordingExplorer(path_to_standarized_data).reader.data

    output_names = ('spike_index', 'waveform', 'probability')

    (spike_index, waveform,
        proba) = detector.predict_recording(data, output_names=output_names)

    detector.predict(x_detect)
    triage.predict(waveform[:, :, :n_neighbors])
    autoencoder.predict(x_ae)


def test_can_use_neural_network_detector(path_to_tests,
                                         path_to_standarized_data):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    data = RecordingsReader(path_to_standarized_data, loader='array').data

    channel_index = make_channel_index(CONFIG.neigh_channels,
                                       CONFIG.geom)

    detection_th = CONFIG.detect.neural_network_detector.threshold_spike
    triage_th = CONFIG.detect.neural_network_triage.threshold_collision
    detection_fname = CONFIG.detect.neural_network_detector.filename
    ae_fname = CONFIG.detect.neural_network_autoencoder.filename
    triage_fname = CONFIG.detect.neural_network_triage.filename

    # instantiate neural networks
    NND = NeuralNetDetector.load(detection_fname, detection_th,
                                 channel_index)
    NNT = NeuralNetTriage.load(triage_fname, triage_th,
                               input_tensor=NND.waveform_tf)
    NNAE = AutoEncoder.load(ae_fname, input_tensor=NND.waveform_tf)

    output_tf = (NNAE.score_tf, NND.spike_index_tf, NNT.idx_clean)

    with tf.Session() as sess:
        NND.restore(sess)
        NNAE.restore(sess)
        NNT.restore(sess)

        rot = NNAE.load_rotation()
        neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)

        neuralnetwork.run_detect_triage_featurize(data, sess, NND.x_tf,
                                                  output_tf,
                                                  neighbors,
                                                  rot)


def test_splitting_in_batches_does_not_affect(path_to_tests,
                                              path_to_standarized_data,
                                              path_to_sample_pipeline_folder):
    yass.set_config(path.join(path_to_tests, 'config_nnet.yaml'))
    CONFIG = yass.read_config()

    PATH_TO_DATA = path_to_standarized_data

    data = RecordingsReader(PATH_TO_DATA, loader='array').data

    with open(path.join(path_to_sample_pipeline_folder, 'preprocess',
                        'standarized.yaml')) as f:
        PARAMS = yaml.load(f)

    channel_index = make_channel_index(CONFIG.neigh_channels,
                                       CONFIG.geom)

    detection_th = CONFIG.detect.neural_network_detector.threshold_spike
    triage_th = CONFIG.detect.neural_network_triage.threshold_collision
    detection_fname = CONFIG.detect.neural_network_detector.filename
    ae_fname = CONFIG.detect.neural_network_autoencoder.filename
    triage_fname = CONFIG.detect.neural_network_triage.filename

    # instantiate neural networks
    NND = NeuralNetDetector.load(detection_fname, detection_th,
                                 channel_index)
    NNT = NeuralNetTriage.load(triage_fname, triage_th,
                               input_tensor=NND.waveform_tf)
    NNAE = AutoEncoder.load(ae_fname, input_tensor=NND.waveform_tf)

    output_tf = (NNAE.score_tf, NND.spike_index_tf, NNT.idx_clean)

    # run all at once
    with tf.Session() as sess:
        # get values of above tensors
        NND.restore(sess)
        NNAE.restore(sess)
        NNT.restore(sess)

        rot = NNAE.load_rotation()
        neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)

        (scores, clear,
         collision) = neuralnetwork.run_detect_triage_featurize(data, sess,
                                                                NND.x_tf,
                                                                output_tf,
                                                                neighbors,
                                                                rot)

    # run in batches - buffer size makes sure we can detect spikes if they
    # appear at the end of any batch
    bp = BatchProcessor(PATH_TO_DATA, PARAMS['dtype'], PARAMS['n_channels'],
                        PARAMS['data_order'], '100KB',
                        buffer_size=CONFIG.spike_size)

    with tf.Session() as sess:
        # get values of above tensors
        NND.restore(sess)
        NNAE.restore(sess)
        NNT.restore(sess)

        rot = NNAE.load_rotation()
        neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)

        res = bp.multi_channel_apply(
            neuralnetwork.run_detect_triage_featurize,
            mode='memory',
            cleanup_function=neuralnetwork.fix_indexes,
            sess=sess,
            x_tf=NND.x_tf,
            output_tf=output_tf,
            rot=rot,
            neighbors=neighbors)

    scores_batch = np.concatenate([element[0] for element in res], axis=0)
    clear_batch = np.concatenate([element[1] for element in res], axis=0)
    collision_batch = np.concatenate([element[2] for element in res], axis=0)

    np.testing.assert_array_equal(clear_batch, clear)
    np.testing.assert_array_equal(collision_batch, collision)
    np.testing.assert_array_equal(scores_batch, scores)
