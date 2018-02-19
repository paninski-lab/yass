import numpy as np
import tensorflow as tf

from yass.util import load_yaml, change_extension


class AutoEncoder(object):
    """
    Class for training and running convolutional neural network detector
    for spike detection
    and autoencoder for feature extraction.

    Attributes:
    -----------
    n_features: int
        number of features to be extracted from the detected waveforms.
    n_input: float
        temporal size of a spike feeded into ae.
    W_ae: tf.Variable
        [n_input, n_features] weight matrix for the autoencoder.
    saver_ae: tf.train.Saver
        saver object for the autoencoder.
    """

    def __init__(self, path_to_ae_model):
        """
        Initializes the attributes for the class NeuralNetDetector.

        Parameters:
        -----------
        path_to_ae_model: str
            location of trained neural net autoencoder
        """

        # add locations as attributes
        self.path_to_ae_model = path_to_ae_model

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
