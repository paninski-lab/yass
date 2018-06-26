import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging

from yass.neuralnetwork.utils import (weight_variable, bias_variable, conv2d,
                                      conv2d_VALID, max_pool)
from yass.util import load_yaml, change_extension
from yass.neuralnetwork.parameter_saver import save_detect_network_params


class NeuralNetDetector(object):
    """
    Class for training and running convolutional neural network detector
    for spike detection
    and autoencoder for feature extraction.

    Parameters
    ----------
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
    threshold: int
        threshold for neural net detection
    channel_index: np.array (n_channels, n_neigh)
        Each row indexes its neighboring channels.
        For example, channel_index[c] is the index of
        neighboring channels (including itself)
        If any value is equal to n_channels, it is nothing but
        a space holder in a case that a channel has less than
        n_neigh neighboring channels

    Attributes
    ----------
    x_tf
        Input layer

    spike_index_tf
        Spike index output layer

    waveform_tf
        Waveform output layer

    probability
        Probability output layer

    """

    def __init__(self, path_to_model, threshold, channel_index):
        """
        Initializes the attributes for the class NeuralNetDetector.

        Parameters:
        -----------
        path_to_model: str
            location of trained neural net detectior
        """
        # add locations as attributes
        if not path_to_model.endswith('.ckpt'):
            path_to_model = path_to_model+'.ckpt'

        self.path_to_model = path_to_model

        # load nn parameter files
        path_to_filters = change_extension(path_to_model, 'yaml')
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

        # placeholder for input recording
        self.x_tf = tf.placeholder("float", [None, None])

        # make spike_index tensorflow tensor
        (self.spike_index_tf_all,
         self.probability) = (self.make_graph(channel_index, threshold))

        # remove edge spike time
        self.spike_index_tf = (self.
                               remove_edge_spikes(self.spike_index_tf_all,
                                                  self.filters_dict['size']))

        # make waveform tensorflow tensor
        size = self.filters_dict['size']
        self.waveform_tf = self.make_waveform_tf(self.spike_index_tf,
                                                 channel_index, size)

    def make_graph(self, channel_index, threshold):
        """Build tensorflow graph with input and two output layers

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
        T = tf.shape(self.x_tf)[0]

        # input tensor into CNN - add one dimension at the beginning and
        # at the end
        x_cnn_tf = tf.expand_dims(tf.expand_dims(self.x_tf, -1), 0)

        # NN structures
        # first temporal layer
        layer1 = tf.nn.relu(conv2d(x_cnn_tf, self.W1) + self.b1)

        # second temporal layer
        layer11 = tf.nn.relu(conv2d(layer1, self.W11) + self.b11)

        # first spatial layer
        zero_added_layer11 = tf.concat((tf.transpose(layer11, [2, 0, 1, 3]),
                                        tf.zeros((1, 1, T, K2))),
                                       axis=0)

        temp = tf.transpose(tf.gather(zero_added_layer11, self.channel_index),
                            [0, 2, 3, 1, 4])

        temp2 = (conv2d_VALID(tf.reshape(temp, [-1, T, nneigh, K2]), self.W2)
                 + self.b2)

        o_layer = tf.transpose(temp2, [2, 1, 0, 3])

        # temporal max
        temporal_max = max_pool(o_layer, [1, 3, 1, 1]) - 1e-8

        # spike index is local maximum crossing a threshold
        spike_index_tf = tf.cast(tf.where(
            tf.logical_and(o_layer[0, :, :, 0] >= temporal_max[0, :, :, 0],
                           o_layer[0, :, :, 0] >
                           np.log(threshold / (1 - threshold)))), 'int32')

        # make probability layer
        probability_layer = tf.sigmoid(o_layer[0, :, :, 0])

        return spike_index_tf, probability_layer

    def remove_edge_spikes(self, spike_index_tf, waveform_length):
        """
        It moves spikes at edge times.

        Parameters
        ----------
        x_tf: tf.tensors (n_observations, n_channels)
            placeholder of recording for running tensorflow

        spike_index_tf: tf tensor (n_spikes, 2)
            a tf tensor holding spike index.
            The first column is time and the second column is the main channel

        waveform_length: int
            temporal length of waveform

        Returns
        -------
        tf tensor (n_spikes, 2)
        """

        R = int((waveform_length-1)/2)
        min_spike_time = R
        max_spike_time = tf.shape(self.x_tf)[0] - R

        idx_middle = tf.logical_and(spike_index_tf[:, 0] > min_spike_time,
                                    spike_index_tf[:, 0] < max_spike_time)

        return tf.boolean_mask(spike_index_tf, idx_middle)

    def make_waveform_tf(self, spike_index_tf,
                         channel_index, waveform_length):
        """
        It produces a tf tensor holding waveforms given recording and spike
        index. It does not hold waveforms on all channels but channels around
        their main channels specified in channel_index

        Parameters
        ----------
        x_tf: tf.tensors (n_observations, n_channels)
            placeholder of recording for running tensorflow

        spike_index_tf: tf tensor (n_spikes, 2)
            a tf tensor holding spike index.
            The first column is time and the second column is the main channel

        channel_index: np.array (n_channels, n_neigh)
            refer above

        waveform_length: int
            temporal length of waveform

        Returns
        -------
        tf tensor (n_spikes, waveform_length, n_neigh)
        """
        # get waveform temporally
        R = int((waveform_length-1)/2)
        spike_time = tf.expand_dims(spike_index_tf[:, 0], -1)
        temporal_index = tf.expand_dims(tf.range(-R, R+1), 0)
        wf_temporal = tf.add(spike_time, temporal_index)

        # get waveform spatially
        nneigh = channel_index.shape[1]
        wf_spatial = tf.gather(channel_index, spike_index_tf[:, 1])

        wf_temporal_expand = tf.expand_dims(
            tf.expand_dims(wf_temporal, -1), -1)
        wf_spatial_expand = tf.expand_dims(
            tf.expand_dims(wf_spatial, 1), -1)

        wf_idx = tf.concat((tf.tile(wf_temporal_expand, (1, 1, nneigh, 1)),
                            tf.tile(wf_spatial_expand,
                                    (1, waveform_length, 1, 1))), 3)

        # temproal length of recording
        T = tf.shape(self.x_tf)[0]
        x_tf_zero_added = tf.concat([self.x_tf, tf.zeros((T, 1))], axis=1)

        return tf.gather_nd(x_tf_zero_added, wf_idx)

    def restore(self, sess):
        """Restore tensor values
        """
        self.saver.restore(sess, self.path_to_model)

    def predict(self, recordings, output_names=('spike_index_tf',)):
        """Make predictions
        """

        output_tensors = [getattr(self, name) for name in output_names]

        with tf.Session() as sess:
            self.restore(sess)

            output = sess.run(output_tensors,
                              feed_dict={self.x_tf: recordings})

        return output

    @classmethod
    def train(cls, x_train, y_train, n_filters, n_iter, n_batch,
              l2_reg_scale, train_step_size, path_to_model):
        """
        Trains the neural network detector for spike detection

        Parameters:
        -----------
        x_train: np.array
            [number of training data, temporal length, number of channels]
            augmented training data consisting of
            isolated spikes, noise and misaligned spikes.
        y_train: np.array
            [number of training data] label for x_train. '1' denotes presence
            of an isolated spike and '0' denotes
            the presence of a noise data or misaligned spike.
        path_to_model: string
            name of the .ckpt to be saved
        """
        # FIXME reuse the make_graph method to avoid repeating code

        logger = logging.getLogger(__name__)

        # get parameters
        n_data, R, C = x_train.shape
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
        _ = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=o_layer, labels=y_tf)
        cross_entropy = tf.reduce_mean(_)

        weights = tf.trainable_variables()

        # regularization term
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)

        regularization = tf.contrib.layers.apply_regularization(l2_regularizer,
                                                                weights)

        regularized_loss = cross_entropy + regularization

        # train step
        train_step = (tf.train.AdamOptimizer(train_step_size)
                        .minimize(regularized_loss))

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

        logger.info('Training detector network...')

        bar = tqdm(total=n_iter)

        with tf.Session() as sess:

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for i in range(0, n_iter):

                # sample n_batch observations from 0, ..., n_data
                idx_batch = np.random.choice(n_data, n_batch, replace=False)

                res = sess.run(
                    [train_step, regularized_loss],
                    feed_dict={
                        x_tf: x_train[idx_batch],
                        y_tf: y_train[idx_batch]
                    })

                bar.update(i + 1)

                # if not i % 100:
                #    logger.info('Loss: %s', res[1])

            saver.save(sess, path_to_model)

            # estimate tp and fp with a sample
            idx_batch = np.random.choice(n_data, n_batch, replace=False)

            output = sess.run(o_layer, feed_dict={x_tf: x_train[idx_batch]})
            y_test = y_train[idx_batch]

            tp = np.mean(output[y_test == 1] > 0)
            fp = np.mean(output[y_test == 0] > 0)

            logger.info('Approximate training true positive rate: ' + str(tp) +
                        ', false positive rate: ' + str(fp))
        bar.close()

        logger.info('Saving detector network parameters...')
        save_detect_network_params(filters=n_filters,
                                   size=x_train.shape[1],
                                   n_neighbors=x_train.shape[2],
                                   output_path=change_extension(path_to_model,
                                                                'yaml'))
