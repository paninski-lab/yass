import numpy as np
import tensorflow as tf

from .utils import (weight_variable, bias_variable, conv2d, conv2d_VALID,
                    max_pool)
from ..util import load_yaml, change_extension


class NeuralNetDetector(object):
    """
        Class for training and running convolutional neural network detector
        for spike detection
        and autoencoder for feature extraction.

        Attributes:
        -----------
        config: configuration object
            configuration object containing the training parameters.
        C: int
            spatial filter size of the spatial convolutional layer.
        R: int
            temporal filter sizes for the temporal convolutional layers.
        K1,K2: int
            number of filters for each convolutional layer.
        W1, W11, W2: tf.Variable
            [temporal_filter_size, spatial_filter_size, input_filter_number,
            ouput_filter_number] weight matrices
            for the covolutional layers.
        b1, b11, b2: tf.Variable
            bias variable for the convolutional layers.
        nFeat: int
            number of features to be extracted from the detected waveforms.
        R: float
            temporal size of a spike.
        W_ae: tf.Variable
            [R, nFeat] weight matrix for the autoencoder.
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

        """

        self.path_to_detector_model = path_to_detector_model
        self.path_to_ae_model = path_to_ae_model

        path_to_filters = change_extension(path_to_detector_model, 'yaml')
        self.filters_dict = load_yaml(path_to_filters)

        R1 = self.filters_dict['size']
        K1, K2 = self.filters_dict['filters']
        C = self.filters_dict['n_neighbors']

        self.W1 = weight_variable([R1, 1, 1, K1])
        self.b1 = bias_variable([K1])

        self.W11 = weight_variable([1, 1, K1, K2])
        self.b11 = bias_variable([K2])

        self.W2 = weight_variable([1, C, K2, 1])
        self.b2 = bias_variable([1])

        # output of ae encoding (1st layer)
        path_to_filters_ae = change_extension(path_to_ae_model, 'yaml')
        ae_dict = load_yaml(path_to_filters_ae)
        n_input = ae_dict['n_input']
        n_features = ae_dict['n_features']
        self.W_ae = tf.Variable(tf.random_uniform(
            (n_input, n_features), -1.0 / np.sqrt(n_input),
            1.0 / np.sqrt(n_input)))

        self.saver_ae = tf.train.Saver({"W_ae": self.W_ae})
        self.saver = tf.train.Saver(
            {"W1": self.W1, "W11": self.W11, "W2": self.W2, "b1": self.b1,
             "b11": self.b11, "b2": self.b2})

    def get_spikes(self, x_tf, T, nneigh, c_idx, temporal_window, th):
        """
            Detects and indexes spikes from the recording. The recording will
            be chopped to minibatches if its temporal length
            exceeds 10000. A spike is detected at [t, c] when the output
            probability of the neural network detector crosses
            the detection threshold at time t and channel c. For temporal
            duplicates within a certain temporal radius,
            the temporal index corresponding to the largest output probability
            is assigned. For spatial duplicates within
            certain neighboring channels, the channel with the highest energy
            is assigned.

            Parameters:
            -----------
            X: np.array
                [number of channels, temporal length] raw recording.

            Returns:
            -----------
            index: np.array
                [number of detected spikes, 3] returned indices for spikes.
                First column corresponds to temporal location;
                second column corresponds to spatial (channel) location.

        """
        # get parameters
        K1, K2 = self.filters_dict['filters']

        # NN structures
        layer1 = tf.nn.relu(conv2d(tf.expand_dims(
            tf.expand_dims(x_tf, -1), 0), self.W1) + self.b1)
        layer11 = tf.nn.relu(conv2d(layer1, self.W11) + self.b11)
        zero_added_layer11 = tf.concat((tf.transpose(layer11, [2, 0, 1, 3]),
                                        tf.zeros((1, 1, T, K2))
                                        ), axis=0)
        temp = tf.transpose(
            tf.gather(zero_added_layer11, c_idx), [0, 2, 3, 1, 4])
        temp2 = conv2d_VALID(tf.reshape(
            temp, [-1, T, nneigh, K2]), self.W2) + self.b2
        o_layer = tf.transpose(temp2, [2, 1, 0, 3])

        temporal_max = max_pool(o_layer, [1, temporal_window, 1, 1])
        local_max_idx = tf.where(tf.logical_and(o_layer[0, :, :, 0] >=
                                                temporal_max[0, :, :, 0],
                                                o_layer[0, :, :, 0] > np.log(
            th/(1-th))
        ))

        return local_max_idx

    def get_score_train(self, x_tf):
        W_ae_conv = tf.expand_dims(tf.expand_dims(self.W_ae, 1), 1)

        scores = conv2d(tf.expand_dims(tf.expand_dims(x_tf, -1), 0),
                        W_ae_conv)

        return tf.squeeze(scores)
