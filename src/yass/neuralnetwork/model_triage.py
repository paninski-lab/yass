import logging
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from yass.neuralnetwork.utils import (weight_variable, bias_variable, conv2d,
                                      conv2d_VALID)
from yass.util import load_yaml, change_extension
from yass.neuralnetwork.parameter_saver import save_triage_network_params


class NeuralNetTriage(object):
    """
        Class for training and running convolutional neural network detector
        for spike detection
        and autoencoder for feature extraction.

        Parameters
        ----------
        path_to_model: str
            location of trained neural net triage
        threshold: float
            Threshold between 0 and 1

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
    """

    def __init__(self, path_to_model, threshold, input_tensor=None):
        if not path_to_model.endswith('.ckpt'):
            path_to_model = path_to_model+'.ckpt'

        self.path_to_model = path_to_model

        # load necessary parameters
        path_to_filters = change_extension(path_to_model, 'yaml')
        self.filters_dict = load_yaml(path_to_filters)
        R1 = self.filters_dict['size']
        K1, K2 = self.filters_dict['filters']
        self.C = self.filters_dict['n_neighbors']

        # initialize and save nn weights
        self.W1 = weight_variable([R1, 1, 1, K1])
        self.b1 = bias_variable([K1])

        self.W11 = weight_variable([1, 1, K1, K2])
        self.b11 = bias_variable([K2])

        self.W2 = weight_variable([1, self.C, K2, 1])
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

        self.idx_clean = self._make_graph(threshold, input_tensor)

    def _make_graph(self, threshold, input_tensor):
        """Builds graph for triage

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
        # input tensor (waveforms)
        if input_tensor:
            self.x_tf = input_tensor
        else:
            self.x_tf = tf.placeholder("float", [None, None, self.C])

        # get parameters
        K1, K2 = self.filters_dict['filters']

        # first layer: temporal feature
        layer1 = tf.nn.relu(conv2d_VALID(tf.expand_dims(self.x_tf, -1),
                                         self.W1) + self.b1)

        # second layer: feataure mapping
        layer11 = tf.nn.relu(conv2d(layer1, self.W11) + self.b11)

        # third layer: spatial convolution
        o_layer = conv2d_VALID(layer11, self.W2) + self.b2

        # thrshold it
        return o_layer[:, 0, 0, 0] > np.log(threshold / (1 - threshold))

    def restore(self, sess):
        """Restore tensor values
        """
        self.saver.restore(sess, self.path_to_model)

    def predict(self, waveforms):
        """Triage waveforms
        """
        with tf.Session() as sess:
            self.restore(sess)

            idx_clean = sess.run(self.idx_clean,
                                 feed_dict={self.x_tf: waveforms})

        return idx_clean

    @classmethod
    def train(cls, x_train, y_train, n_filters, n_iter, n_batch,
              l2_reg_scale, train_step_size, path_to_model):
        """
        Trains the triage network

        Parameters:
        -----------
        x_train: np.array
            [number of data, temporal length, number of channels] training data
            for the triage network.
        y_train: np.array
            [number of data] training label for the triage network.
        path_to_model: string
            name of the .ckpt to be saved.
        """
        logger = logging.getLogger(__name__)

        if not path_to_model.endswith('.ckpt'):
            path_to_model = path_to_model+'.ckpt'

        # get parameters
        ndata, R, C = x_train.shape
        K1, K2 = n_filters

        # x and y input tensors
        x_tf = tf.placeholder("float", [n_batch, R, C])
        y_tf = tf.placeholder("float", [n_batch])

        # first layer: temporal feature
        W1 = weight_variable([R, 1, 1, K1])
        b1 = bias_variable([K1])
        layer1 = tf.nn.relu(conv2d_VALID(tf.expand_dims(x_tf, -1), W1) + b1)

        # second layer: feataure mapping
        W11 = weight_variable([1, 1, K1, K2])
        b11 = bias_variable([K2])
        layer11 = tf.nn.relu(conv2d(layer1, W11) + b11)

        # third layer: spatial convolution
        W2 = weight_variable([1, C, K2, 1])
        b2 = bias_variable([1])
        o_layer = tf.squeeze(conv2d_VALID(layer11, W2) + b2)

        # cross entropy
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=o_layer,
                                                    labels=y_tf))

        # regularization term
        weights = tf.trainable_variables()
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        regularization_penalty = tf.contrib.layers.apply_regularization(
            l2_regularizer, weights)
        regularized_loss = cross_entropy + regularization_penalty

        # train step
        train_step = tf.train.AdamOptimizer(train_step_size).minimize(
            regularized_loss)

        # saver
        saver = tf.train.Saver({
            "W1": W1,
            "W11": W11,
            "W2": W2,
            "b1": b1,
            "b11": b11,
            "b2": b2
        })

        ############
        # training #
        ############

        logger.info('Training triage network...')

        bar = tqdm(total=n_iter)

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for i in range(0, n_iter):

                idx_batch = np.random.choice(ndata, n_batch, replace=False)
                sess.run(
                    train_step,
                    feed_dict={
                        x_tf: x_train[idx_batch],
                        y_tf: y_train[idx_batch]
                    })
                bar.update(i + 1)

            saver.save(sess, path_to_model)

            idx_batch = np.random.choice(ndata, n_batch, replace=False)
            output = sess.run(o_layer, feed_dict={x_tf: x_train[idx_batch]})
            y_test = y_train[idx_batch]
            tp = np.mean(output[y_test == 1] > 0)
            fp = np.mean(output[y_test == 0] > 0)

            logger.info('Approximate training true positive rate: ' + str(tp) +
                        ', false positive rate: ' + str(fp))
        bar.close()

        logger.info('Saving triage network parameters...')
        save_triage_network_params(filters=n_filters,
                                   size=x_train.shape[1],
                                   n_neighbors=x_train.shape[2],
                                   output_path=change_extension(path_to_model,
                                                                'yaml'))
