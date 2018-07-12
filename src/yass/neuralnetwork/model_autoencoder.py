import logging
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from yass.util import load_yaml, change_extension, dict2yaml
from yass.neuralnetwork.model import Model


class AutoEncoder(Model):
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

    def __init__(self, path_to_model, waveform_length, n_features,
                 input_tensor=None):
        """
        Initializes the attributes for the class NeuralNetDetector.

        Parameters:
        -----------
        path_to_model: str
            location of trained neural net autoencoder
        """
        self.path_to_model = path_to_model
        self.waveform_length = waveform_length
        self.n_features = n_features

        W_ae = tf.Variable(
            tf.random_uniform((waveform_length, n_features),
                              -1.0 / np.sqrt(waveform_length),
                              1.0 / np.sqrt(waveform_length)))
        self.vars_dict = {"W_ae": W_ae}
        self.saver = tf.train.Saver(self.vars_dict)

        # make score tensorflow tensor from waveform
        self.score_tf = self._make_graph(input_tensor)

    @classmethod
    def load(cls, path_to_model, input_tensor=None):

        if not path_to_model.endswith('.ckpt'):
            path_to_model = path_to_model+'.ckpt'

        # load parameter of autoencoder
        path_to_params = change_extension(path_to_model, 'yaml')
        params = load_yaml(path_to_params)

        return cls(path_to_model, params['waveform_length'],
                   params['n_features'], input_tensor)

    def _make_graph(self, input_tensor):
        """
        Make a tensorflow tensor that outputs scores

        Parameters
        -----------
        input_tensor: tf tensor (n_spikes, n_temporal_size, n_neigh)
            tensorflow tensor that contains waveforms of spikes

        Returns
        -------
        score_tf: tf tensor (n_spikes, n_features, n_neigh)
            tensorflow tensor that produces scores
        """
        # input tensor (waveforms)
        if input_tensor is None:
            self.x_tf = tf.placeholder("float", [None, self.waveform_length])
            score_tf = tf.matmul(self.x_tf, self.vars_dict['W_ae'])
        else:
            self.x_tf = input_tensor
            # transpose to the expected input and flatten
            reshaped_wf = tf.reshape(tf.transpose(self.x_tf, [0, 2, 1]),
                                     [-1, self.waveform_length])
            n_neigh = tf.shape(self.x_tf)[2]

            mult = tf.matmul(reshaped_wf, self.vars_dict['W_ae'])
            mult_reshaped = tf.reshape(mult, [-1, n_neigh, self.n_features])
            score_tf = tf.transpose(mult_reshaped, [0, 2, 1])

        return score_tf

    def load_rotation(self):
        """
        Load neural network rotation matrix
        """

        with tf.Session() as sess:
            self.saver.restore(sess, self.path_to_model)
            rotation = sess.run(self.vars_dict['W_ae'])

        return rotation

    def restore(self, sess):
        """Restore tensor values
        """
        self.saver.restore(sess, self.path_to_model)

    def predict(self, waveforms):
        """Apply autoencoder
        """
        n_waveforms, waveform_length = waveforms.shape
        self._validate_dimensions(waveform_length)

        with tf.Session() as sess:
            self.restore(sess)

            scores = sess.run(self.score_tf,
                              feed_dict={self.x_tf: waveforms})

        return scores

    def fit(self, x_train):
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
        # FIXME: y_ae no longer used

        logger = logging.getLogger(__name__)

        # parameters
        n_data, waveform_length = x_train.shape

        self._validate_dimensions(waveform_length)

        pca = PCA(n_components=self.n_features).fit(x_train)

        self.vars_dict['W_ae'] = (tf.Variable((pca.components_.T)
                                              .astype('float32')))

        # saver
        saver = tf.train.Saver(self.vars_dict)

        ############
        # training #
        ############

        logger.info('Training autoencoder network...')

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.save(sess, self.path_to_model)

        dict2yaml(output_path=change_extension(self.path_to_model,
                                               'yaml'),
                  waveform_length=self.waveform_length,
                  n_features=self.n_features)
