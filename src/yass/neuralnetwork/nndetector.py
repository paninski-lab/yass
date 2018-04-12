import numpy as np
import tensorflow as tf

from yass.neuralnetwork.utils import (weight_variable, bias_variable, conv2d,
                                      conv2d_VALID, max_pool)
from yass.util import load_yaml, change_extension


class NeuralNetDetector(object):
    """
    Class for training and running convolutional neural network detector
    for spike detection
    and autoencoder for feature extraction.

    Attributes:
    -----------
    C: int
        spatial filter size of the spatial convolutional layer.
    R1: int
        temporal filter sizes for the temporal convolutional layers.
    K1,K2: int
        number of filters for each convolutional layer.
    W1, W11, W2: tf.Variable
        [temporal_filter_size, spatial_filter_size, input_filter_number,
        ouput_filter_number] weight matrices
        for the covolutional layers.
    b1, b11, b2: tf.Variable
        bias variable for the convolutional layers.
    saver: tf.train.Saver
        saver object for the neural network detector.
    """

    def __init__(self, path_to_detector_model):
        """
        Initializes the attributes for the class NeuralNetDetector.

        Parameters:
        -----------
        path_to_detector_model: str
            location of trained neural net detectior
        """

        # add locations as attributes
        self.path_to_detector_model = path_to_detector_model

        # load nn parameter files
        path_to_filters = change_extension(path_to_detector_model, 'yaml')
        self.filters_dict = load_yaml(path_to_filters)

        # initialize neural net weights and add as attributes
        R1 = self.filters_dict['size']
        K1, K2 = self.filters_dict['filters']
        C = self.filters_dict['n_neighbors']

        self.W1 = weight_variable([R1, 1, 1, K1])
        self.b1 = bias_variable([K1])

        self.W11 = weight_variable([1, 1, K1, K2])
        self.b11 = bias_variable([K2])

        self.W2 = weight_variable([1, C, K2, 1])
        self.b2 = bias_variable([1])

        # create saver variables
        self.saver = tf.train.Saver({
            "W1": self.W1,
            "W11": self.W11,
            "W2": self.W2,
            "b1": self.b1,
            "b11": self.b11,
            "b2": self.b2
        })

    def make_detection_tf_tensors(self, x_tf, channel_index, threshold):
        """
        Make a tensorflow tensor that outputs spike index

        Parameters
        -----------
        x_tf: tf.tensors (n_observations, n_channels)
            placeholder of recording for running tensorflow

        channel_index: np.array (n_channels, n_neigh)
            Each row indexes its neighboring channels.
            For example, channel_index[c] is the index of
            neighboring channels (including itself)
            If any value is equal to n_channels, it is nothing but
            a space holder in a case that a channel has less than
            n_neigh neighboring channels

        threshold: int
            threshold on a probability to determine
            location of spikes

        Returns
        -------
        spike_index_tf: tf tensor (n_spikes, 2)
            tensorflow tensor that produces spike_index
        """

        # get parameters
        K1, K2 = self.filters_dict['filters']
        nneigh = self.filters_dict['n_neighbors']

        # save neighbor channel index
        self.channel_index = channel_index[:, :nneigh]

        # Temporal shape of input
        T = tf.shape(x_tf)[0]

        # input tensor into CNN
        x_cnn_tf = tf.expand_dims(tf.expand_dims(x_tf, -1), 0)

        # NN structures
        # first temporal layer
        layer1 = tf.nn.relu(conv2d(x_cnn_tf, self.W1) + self.b1)

        # second temporal layer
        layer11 = tf.nn.relu(conv2d(layer1, self.W11) + self.b11)

        # first spatial layer
        zero_added_layer11 = tf.concat(
            (tf.transpose(layer11, [2, 0, 1, 3]), tf.zeros((1, 1, T, K2))),
            axis=0)
        temp = tf.transpose(
            tf.gather(zero_added_layer11, self.channel_index), [0, 2, 3, 1, 4])
        temp2 = conv2d_VALID(tf.reshape(temp, [-1, T, nneigh, K2]),
                             self.W2) + self.b2

        # output layer
        o_layer = tf.transpose(temp2, [2, 1, 0, 3])

        # temporal max
        temporal_max = max_pool(o_layer, [1, 3, 1, 1]) - 1e-8

        # spike index is local maximum crossing a threshold
        spike_index_tf = tf.cast(tf.where(
            tf.logical_and(o_layer[0, :, :, 0] >= temporal_max[0, :, :, 0],
                           o_layer[0, :, :, 0] >
                           np.log(threshold / (1 - threshold)))), 'int32')

        return spike_index_tf

    def make_o_layer_tf_tensors(self, x_tf, channel_index):
        """
        Make a tensorflow tensor that outputs spike index

        Parameters
        -----------
        x_tf: tf.tensors (n_observations, n_channels)
            placeholder of recording for running tensorflow

        channel_index: np.array (n_channels, n_neigh)
            Each row indexes its neighboring channels.
            For example, channel_index[c] is the index of
            neighboring channels (including itself)
            If any value is equal to n_channels, it is nothing but
            a space holder in a case that a channel has less than
            n_neigh neighboring channels

        Returns
        -------
        output_tf: tf tensor (n_observations, n_channels)
            tensorflow tensor that produces spike_index
        """

        # get parameters
        K1, K2 = self.filters_dict['filters']
        nneigh = self.filters_dict['n_neighbors']

        # save neighbor channel index
        self.channel_index = channel_index[:, :nneigh]

        # Temporal shape of input
        T = tf.shape(x_tf)[0]

        # input tensor into CNN
        x_cnn_tf = tf.expand_dims(tf.expand_dims(x_tf, -1), 0)

        # NN structures
        # first temporal layer
        layer1 = tf.nn.relu(conv2d(x_cnn_tf, self.W1) + self.b1)

        # second temporal layer
        layer11 = tf.nn.relu(conv2d(layer1, self.W11) + self.b11)

        # first spatial layer
        zero_added_layer11 = tf.concat(
            (tf.transpose(layer11, [2, 0, 1, 3]), tf.zeros((1, 1, T, K2))),
            axis=0)
        temp = tf.transpose(
            tf.gather(zero_added_layer11, self.channel_index),
            [0, 2, 3, 1, 4])
        temp2 = conv2d_VALID(tf.reshape(temp, [-1, T, nneigh, K2]),
                             self.W2) + self.b2

        # output layer
        # o_layer: [1, temporal, spatial, 1]
        o_layer = tf.transpose(temp2, [2, 1, 0, 3])[0, :, :, 0]
        output_tf = tf.sigmoid(o_layer)

        return output_tf
