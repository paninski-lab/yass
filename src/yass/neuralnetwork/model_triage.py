try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

import warnings
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import trange

from yass.neuralnetwork.utils import (weight_variable, bias_variable, conv2d,
                                      conv2d_VALID)
from yass.util import load_yaml, change_extension
from yass.neuralnetwork.model import Model


class NeuralNetTriage(Model):
    """Convolutional Neural Network for spike detection

    Parameters
    ----------
    path_to_model: str
        Where to save the trained model

    threshold: float
        Threshold between 0 and 1, values higher than the threshold are
        considered spikes

    input_tensor

    Attributes
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
    detector: NeuralNetDetector
        Instance of detector
    threshold: int
        threshold for neural net triage
    CLEAN: int
        Label assigned to the clean spike class (1)
    COLLIDED: int
        Label assigned to the collided spike class (0)

    """

    CLEAN = 1
    COLLIDED = 0

    def __init__(self, path_to_model, filters_size,
                 waveform_length, n_neighbors, threshold,
                 n_iter=50000, n_batch=512, l2_reg_scale=0.00000005,
                 train_step_size=0.001, input_tensor=None,
                 load_test_set=False):

        self.logger = logging.getLogger(__name__)

        if input_tensor is not None:
            if n_neighbors != input_tensor.shape[2]:

                warnings.warn('Network n_neighbors ({}) does not match '
                              'n_neighbors on input_tensor ({}), using '
                              'only the first n_neighbors from the '
                              'input_tensor'.format(n_neighbors,
                                                    input_tensor.shape[2]))

        self.path_to_model = path_to_model
        self.model_name = Path(path_to_model).name.replace('.ckpt', '')

        self.filters_size = filters_size
        self.n_neighbors = n_neighbors
        self.waveform_length = waveform_length

        self.threshold = threshold
        self.n_batch = n_batch
        self.l2_reg_scale = l2_reg_scale
        self.train_step_size = train_step_size
        self.n_iter = n_iter

        self.idx_clean = self._make_graph(threshold, input_tensor,
                                          filters_size,
                                          waveform_length,
                                          n_neighbors)

        if load_test_set:
            self._load_test_set()

    @classmethod
    def load(cls, path_to_model, threshold, input_tensor=None,
             load_test_set=False):
        """Load a model from a file
        """
        if not path_to_model.endswith('.ckpt'):
            path_to_model = path_to_model+'.ckpt'

        # load necessary parameters
        path_to_params = change_extension(path_to_model, 'yaml')
        params = load_yaml(path_to_params)

        return cls(path_to_model=path_to_model,
                   filters_size=params['filters_size'],
                   waveform_length=params['waveform_length'],
                   n_neighbors=params['n_neighbors'],
                   threshold=threshold,
                   input_tensor=input_tensor, load_test_set=load_test_set)

    @classmethod
    def _make_network(cls, input_tensor, filters_size, waveform_length,
                      n_neighbors):
        """Mates tensorflow network, from first layer to output layer
        """
        K1, K2 = filters_size

        # initialize and save nn weights
        W1 = weight_variable([waveform_length, 1, 1, K1])
        b1 = bias_variable([K1])

        W11 = weight_variable([1, 1, K1, K2])
        b11 = bias_variable([K2])

        W2 = weight_variable([1, n_neighbors, K2, 1])
        b2 = bias_variable([1])

        # first layer: temporal feature
        layer1 = tf.nn.relu(conv2d_VALID(tf.expand_dims(input_tensor, -1),
                                         W1) + b1)

        # second layer: feataure mapping
        layer11 = tf.nn.relu(conv2d(layer1, W11) + b11)

        # third layer: spatial convolution
        o_layer = conv2d_VALID(layer11, W2) + b2

        vars_dict = {"W1": W1, "W11": W11, "W2": W2, "b1": b1, "b11": b11,
                     "b2": b2}

        return o_layer, vars_dict

    def _make_graph(self, threshold, input_tensor, filters_size,
                    waveform_length, n_neighbors):
        """Builds graph for triage

        Parameters:
        -----------
        input_tensor: tf tensor (n_spikes, n_temporal_length, n_neighbors)
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
        # input tensor (waveforms)
        if input_tensor is None:
            self.x_tf = tf.placeholder("float", [None, None, n_neighbors])
        else:
            self.x_tf = input_tensor

        (self.o_layer,
            vars_dict) = NeuralNetTriage._make_network(self.x_tf,
                                                       filters_size,
                                                       waveform_length,
                                                       n_neighbors)

        self.saver = tf.train.Saver(vars_dict)

        # thrshold it
        return self.o_layer[:, 0, 0, 0] > np.log(threshold / (1 - threshold))

    def restore(self, sess):
        """Restore tensor values
        """
        self.logger.debug('Restoring tensorflow session from: %s',
                          self.path_to_model)
        self.saver.restore(sess, self.path_to_model)

    def predict(self, waveforms):
        """Triage waveforms
        """
        _, waveform_length, n_neighbors = waveforms.shape

        # self._validate_dimensions(waveform_length, n_neighbors)

        with tf.Session() as sess:
            self.restore(sess)

            idx_clean = sess.run(self.idx_clean,
                                 feed_dict={self.x_tf: waveforms})

        return idx_clean

    def fit(self, x_train, y_train, test_size=0.3, save_test_set=False):
        """Trains the triage network

        Parameters
        ----------
        x_train: np.array
            [number of data, temporal length, number of channels] training data
            for the triage network.
        y_train: np.array
            [number of data] training label for the triage network.
        test_size: float, optional
            Proportion of the training set to be used, data is shuffled before
            splitting, defaults to 0.3

        Returns
        -------
        dict
            Dictionary with network parameters and metrics

        Notes
        -----
        Size is determined but the second dimension in x_train
        """
        #####################
        # Splitting dataset #
        #####################

        (self.x_train, self.x_test,
         self.y_train, self.y_test) = train_test_split(x_train, y_train,
                                                       test_size=test_size)

        # get parameters
        n_data, waveform_length_train, n_neighbors_train = self.x_train.shape

        self._validate_dimensions(waveform_length_train, n_neighbors_train)

        # x and y input tensors
        x_tf = tf.placeholder("float", [None, self.waveform_length,
                                        self.n_neighbors])
        y_tf = tf.placeholder("float", [None])

        o_layer, vars_dict = (NeuralNetTriage
                              ._make_network(x_tf,
                                             self.filters_size,
                                             self.waveform_length,
                                             self.n_neighbors))
        logits = tf.squeeze(o_layer)

        # cross entropy
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                    labels=y_tf))

        # regularization term
        weights = tf.trainable_variables()
        l2_regularizer = (tf.contrib.layers
                            .l2_regularizer(scale=self.l2_reg_scale))
        regularization_penalty = tf.contrib.layers.apply_regularization(
            l2_regularizer, weights)
        regularized_loss = cross_entropy + regularization_penalty

        # train step
        train_step = tf.train.AdamOptimizer(self.train_step_size).minimize(
            regularized_loss)

        # saver
        saver = tf.train.Saver(vars_dict)

        ############
        # training #
        ############

        self.logger.debug('Training triage network...')

        with tf.Session() as sess:

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            pbar = trange(self.n_iter)

            for i in pbar:

                idx_batch = np.random.choice(n_data, self.n_batch,
                                             replace=False)

                res = sess.run([train_step, regularized_loss],
                               feed_dict={x_tf: self.x_train[idx_batch],
                                          y_tf: self.y_train[idx_batch]})

                if i % 100 == 0:
                    # compute validation loss and metrics
                    output = sess.run({'val loss': regularized_loss},
                                      feed_dict={x_tf: self.x_test,
                                                 y_tf: self.y_test})
                    pbar.set_description('Tr loss: %s, '
                                         'Val loss: %s' % (res[1],
                                                           output['val loss']))

            self.logger.debug('Saving network: %s', self.path_to_model)
            saver.save(sess, self.path_to_model)

        path_to_params = change_extension(self.path_to_model, 'yaml')
        self.logger.debug('Saving network parameters: %s', path_to_params)

        params = dict(filters_size=self.filters_size,
                      waveform_length=self.waveform_length,
                      n_neighbors=self.n_neighbors,
                      name=self.model_name)

        # compute metrics (print them and return them)
        metrics = self._evaluate()

        params.update(metrics)

        # save parameters to disk
        self._save_params(path=path_to_params, params=params)

        if save_test_set:
            self._save_test_set()

        return params
