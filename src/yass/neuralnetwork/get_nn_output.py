"""
Detection pipeline
"""

import os.path

import tensorflow as tf
from yass import read_config
from yass.batch import BatchProcessor
from yass.geometry import make_channel_index
from yass.neuralnetwork import NeuralNetDetector


# TODO: missing parameters docs
def get_o_layer(standarized_path, standarized_params,
                output_directory='tmp/',
                output_dtype='float32', output_filename='o_layer.bin',
                if_file_exists='skip', save_partial_results=False):
    """Get the output of NN detector instead of outputting the spike index
    """

    CONFIG = read_config()
    channel_index = make_channel_index(CONFIG.neigh_channels,
                                       CONFIG.geom, 1)

    x_tf = tf.placeholder("float", [None, None])

    # load Neural Net's
    detection_fname = CONFIG.detect.neural_network_detector.filename
    detection_th = CONFIG.detect.neural_network_detector.threshold_spike
    NND = NeuralNetDetector(detection_fname)
    o_layer_tf = NND.make_o_layer_tf_tensors(x_tf,
                                             channel_index,
                                             detection_th)

    bp = BatchProcessor(standarized_path,
                        standarized_params['dtype'],
                        standarized_params['n_channels'],
                        standarized_params['data_format'],
                        CONFIG.resources.max_memory,
                        buffer_size=CONFIG.spike_size)

    TMP = os.path.join(CONFIG.data.root_folder, output_directory)
    _output_path = os.path.join(TMP, output_filename)
    (o_path,
     o_params) = bp.multi_channel_apply(_get_o_layer,
                                        mode='disk',
                                        cleanup_function=fix_indexes,
                                        output_path=_output_path,
                                        cast_dtype=output_dtype,
                                        x_tf=x_tf,
                                        o_layer_tf=o_layer_tf,
                                        NND=NND)

    return o_path, o_params


def _get_o_layer(recordings, x_tf, o_layer_tf, NND):
    """Detect spikes using a neural network

    Parameters
    ----------
    recordings: numpy.ndarray (n_observations, n_channels)
        Neural recordings

    x_tf: tf.tensors (n_observations, n_channels)
        placeholder of recording for running tensorflow

    output_tf: tuple of tf.tensors
        a tuple of tensorflow tensors that produce score, spike_index_clear,
        and spike_index_collision

    Returns
    -------
    scores: numpy.ndarray (n_clear_spikes, n_features, n_neigh)
        3D array with the scores for the clear spikes, first dimension is
        the number of spikes, second is the nymber of features and third the
        number of neighboring channels

    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    spike_index_collision: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for collided spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)
    """

    # get values of above tensors
    with tf.Session() as sess:
        NND.saver.restore(sess, NND.path_to_detector_model)
        o_layer = sess.run(
            o_layer_tf, feed_dict={x_tf: recordings})

    return o_layer


def fix_indexes(res, idx_local, idx, buffer_size):
    """Fixes indexes from detected spikes in batches

    Parameters
    ----------
    res: tuple
        A result from the butterworth
    idx_local: slice
        A slice object indicating the indices for the data (excluding buffer)
    idx: slice
        A slice object indicating the absolute location of the data
    buffer_size: int
        Buffer size
    """

    # get limits for the data (exlude indexes that have buffer data)
    data_start = idx_local[0].start
    data_end = idx_local[0].stop

    return res[data_start:data_end]
