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
    n_features: int
        number of features to be extracted from the detected waveforms.
    n_input: float
        temporal size of a spike feeded into ae.
    W_ae: tf.Variable
        [n_input, n_features] weight matrix for the autoencoder.
    saver_ae: tf.train.Saver
        saver object for the autoencoder.
    saver: tf.train.Saver
        saver object for the neural network detector.
    """

    def __init__(self, path_to_detector_model, path_to_ae_model):
        """
        Initializes the attributes for the class NeuralNetDetector.

        Parameters:
        -----------
        path_to_detector_model: str
            location of trained neural net detectior

        path_to_ae_model: str
            location of trained neural net autoencoder
        """

        # add locations as attributes
        self.path_to_detector_model = path_to_detector_model
        self.path_to_ae_model = path_to_ae_model

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

        # load parameter of autoencoder
        path_to_filters_ae = change_extension(path_to_ae_model, 'yaml')
        self.ae_dict = load_yaml(path_to_filters_ae)
        n_input = self.ae_dict['n_input']
        n_features = self.ae_dict['n_features']
        # initialize autoencoder weight
        self.W_ae = tf.Variable(
            tf.random_uniform((n_input, n_features), -1.0 / np.sqrt(n_input),
                              1.0 / np.sqrt(n_input)))

        # create saver variables
        self.saver_ae = tf.train.Saver({"W_ae": self.W_ae})
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
        nneigh = channel_index.shape[1]

        # save neighbor channel index
        self.channel_index = channel_index

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
            tf.gather(zero_added_layer11, channel_index), [0, 2, 3, 1, 4])
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

    def make_score_tf_tensor(self, waveform_tf):
        """
        Make a tensorflow tensor that outputs scores

        Parameters
        -----------
        waveform_tf: tf tensor (n_spikes, n_temporal_size, n_neigh)
            tensorflow tensor that contains waveforms of spikes

        Returns
        -------
        score_tf: tf tensor (n_spikes, n_features, n_neigh)
            tensorflow tensor that produces scores
        """

        n_input = self.ae_dict['n_input']
        n_features = self.ae_dict['n_features']
        nneigh_tf = tf.shape(waveform_tf)[2]

        reshaped_wf = tf.reshape(tf.transpose(waveform_tf, [0, 2, 1]),
                                 [-1, n_input])
        score_tf = tf.transpose(tf.reshape(tf.matmul(reshaped_wf, self.W_ae),
                                           [-1, nneigh_tf, n_features]),
                                [0, 2, 1])

        return score_tf

    def load_rotation(self):
        """
        Load neural network rotation matrix
        """

        with tf.Session() as sess:
            self.saver_ae.restore(sess, self.path_to_ae_model)
            rotation = sess.run(self.W_ae)

        return rotation
