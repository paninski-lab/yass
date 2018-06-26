import logging
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from yass.util import load_yaml, change_extension
from yass.neuralnetwork.parameter_saver import save_ae_network_params


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
    saver: tf.train.Saver
        saver object for the autoencoder.
    detector: NeuralNetDetector
        Instance of detector
    """

    def __init__(self, path_to_model, detector):
        """
        Initializes the attributes for the class NeuralNetDetector.

        Parameters:
        -----------
        path_to_model: str
            location of trained neural net autoencoder
        """
        if not path_to_model.endswith('.ckpt'):
            path_to_model = path_to_model+'.ckpt'

        self.path_to_model = path_to_model

        # load parameter of autoencoder
        path_to_filters_ae = change_extension(path_to_model, 'yaml')
        self.ae_dict = load_yaml(path_to_filters_ae)
        n_input = self.ae_dict['n_input']
        n_features = self.ae_dict['n_features']

        # initialize autoencoder weight
        self.W_ae = tf.Variable(
            tf.random_uniform((n_input, n_features), -1.0 / np.sqrt(n_input),
                              1.0 / np.sqrt(n_input)))

        # create saver variables
        self.saver = tf.train.Saver({"W_ae": self.W_ae})

        # make score tensorflow tensor from waveform
        self.score_tf = self.make_score_tf_tensor(detector.waveform_tf)

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
            self.saver.restore(sess, self.path_to_model)
            rotation = sess.run(self.W_ae)

        return rotation

    def restore(self, sess):
        """Restore tensor values
        """
        self.saver.restore(sess, self.path_to_model)

    @classmethod
    def train(cls, x_train, y_train, n_features, n_iter, n_batch,
              train_step_size, path_to_model):
        """
        Trains the autoencoder for feature extraction

        Parameters:
        -----------
        x_train: np.array
            [number of training data, temporal length] noisy isolated spikes
            for training the autoencoder.
        y_train: np.array
            [number of training data, temporal length] clean (denoised)
            isolated spikes as labels.
        path_to_model: string
            name of the .ckpt to be saved.
        """
        logger = logging.getLogger(__name__)

        if not path_to_model.endswith('.ckpt'):
            path_to_model = path_to_model+'.ckpt'

        # parameters
        n_data, n_input = x_train.shape

        pca = PCA(n_components=n_features).fit(x_train)

        # encoding
        W_ae = tf.Variable((pca.components_.T).astype('float32'))

        # saver
        saver = tf.train.Saver({"W_ae": W_ae})

        ############
        # training #
        ############

        logger.info('Training autoencoder network...')

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.save(sess, path_to_model)

        save_ae_network_params(n_input=x_train.shape[1],
                               n_features=n_features,
                               output_path=change_extension(path_to_model,
                                                            'yaml'))
