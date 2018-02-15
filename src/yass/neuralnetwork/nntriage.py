import tensorflow as tf
import numpy as np

from yass.neuralnetwork.utils import (weight_variable, bias_variable, conv2d,
                                      conv2d_VALID)
from yass.util import load_yaml, change_extension


class NeuralNetTriage(object):
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

    def __init__(self, path_to_triage_model):
        """
            Initializes the attributes for the class NeuralNetTriage.

            Parameters:
            -----------
            path_to_detector_model: str
                location of trained neural net triage
        """
        # save path to the model as an attribute
        self.path_to_triage_model = path_to_triage_model

        # load necessary parameters
        path_to_filters = change_extension(path_to_triage_model, 'yaml')
        self.filters_dict = load_yaml(path_to_filters)
        R1 = self.filters_dict['size']
        K1, K2 = self.filters_dict['filters']
        C = self.filters_dict['n_neighbors']

        # initialize and save nn weights
        self.W1 = weight_variable([R1, 1, 1, K1])
        self.b1 = bias_variable([K1])

        self.W11 = weight_variable([1, 1, K1, K2])
        self.b11 = bias_variable([K2])

        self.W2 = weight_variable([1, C, K2, 1])
        self.b2 = bias_variable([1])

        # initialize savers
        self.saver = tf.train.Saver({
            "W1": self.W1,
            "W11": self.W11,
            "W2": self.W2,
            "b1": self.b1,
            "b11": self.b11,
            "b2": self.b2
        })

    def triage_wf(self, wf_tf, threshold):
        """
            Run neural net triage on given spike waveforms

            Parameters:
            -----------
            wf_tf: tf tensor (n_spikes, n_temporal_length, n_neigh)
                tf tensor that produces spikes waveforms

            threshold: int
                threshold used on a probability obtained after nn to determine
                whether it is a clear spike

            Returns:
            -----------
            tf tensor (n_spikes,)
                a boolean tensorflow tensor that produces indices of
                clear spikes
        """
        # get parameters
        K1, K2 = self.filters_dict['filters']

        # first layer: temporal feature
        layer1 = tf.nn.relu(conv2d_VALID(tf.expand_dims(wf_tf, -1),
                                         self.W1) + self.b1)

        # second layer: feataure mapping
        layer11 = tf.nn.relu(conv2d(layer1, self.W11) + self.b11)

        # third layer: spatial convolution
        o_layer = conv2d_VALID(layer11, self.W2) + self.b2

        # thrshold it
        return o_layer[:, 0, 0, 0] > np.log(threshold / (1 - threshold))
