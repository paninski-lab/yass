import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging

from yass.neuralnetwork.utils import (weight_variable, bias_variable, conv2d,
                                      conv2d_VALID, max_pool)
from yass.util import load_yaml, change_extension
from yass.neuralnetwork.parameter_saver import save_detect_network_params


# FIXME: missing documentation, how does changing the step parameter in
# the passed channel_index chages the detector behavior?
class NeuralNetDetector(object):
    """
    Class for training and running convolutional neural network detector
    for spike detection

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
            Each row indexes its neighboring channels. For example,
            channel_index[c] is the index of neighboring channels (including
            itself) If any value is equal to n_channels, it is nothing but
            a placeholder in a case that a channel has less than n_neigh
            neighboring channels
    """

    def __init__(self, path_to_model, filters_size, waveform_length,
                 n_neighbors, threshold, channel_index, n_iter=50000,
                 n_batch=512, l2_reg_scale=0.00000005,
                 train_step_size=0.001):
        """
        Initializes the attributes for the class NeuralNetDetector.

        Parameters:
        -----------
        path_to_model: str
            location of trained neural net detectior
        """
        self.path_to_model = path_to_model

        self.filters_size = filters_size
        self.n_neighbors = n_neighbors
        self.waveform_length = waveform_length

        self.threshold = threshold
        self.n_batch = n_batch
        self.l2_reg_scale = l2_reg_scale
        self.train_step_size = train_step_size
        self.n_iter = n_iter

        # make spike_index tensorflow tensor
        (self.x_tf,
         self.spike_index_tf,
         self.probability_tf,
         self.waveform_tf,
         vars_dict) = NeuralNetDetector._make_graph(threshold,
                                                    channel_index,
                                                    waveform_length,
                                                    filters_size,
                                                    n_neighbors)

        # create saver variables
        self.saver = tf.train.Saver(vars_dict)

    @classmethod
    def load(cls, path_to_model, threshold, channel_index):

        if not path_to_model.endswith('.ckpt'):
            path_to_model = path_to_model+'.ckpt'

        # load nn parameter files
        path_to_params = change_extension(path_to_model, 'yaml')
        params = load_yaml(path_to_params)

        return cls(path_to_model, params['filters_size'],
                   params['waveform_length'], params['n_neighbors'],
                   threshold, channel_index)

    @classmethod
    def _make_network(cls, input_layer, waveform_length, filters_size,
                      n_neigh, padding):
        K1, K2 = filters_size

        W1 = weight_variable([waveform_length, 1, 1, K1])
        b1 = bias_variable([K1])

        W11 = weight_variable([1, 1, K1, K2])
        b11 = bias_variable([K2])

        W2 = weight_variable([1, n_neigh, K2, 1])
        b2 = bias_variable([1])

        vars_dict = {"W1": W1, "W11": W11, "W2": W2, "b1": b1, "b11": b11,
                     "b2": b2}

        # first temporal layer
        # FIXME: old training code was using conv2d_VALID, old graph building
        # for prediction was using conv2d, that's why I need to add the
        # padding parameter, otherwise it breaks. we need to fix it
        layer1 = tf.nn.relu(conv2d(input_layer, W1, padding) + b1)

        # second temporal layer
        layer11 = tf.nn.relu(conv2d(layer1, W11) + b11)

        return vars_dict, layer11

    @classmethod
    def _make_graph(cls, threshold, channel_index, waveform_length,
                    filters_size, n_neigh):
        """Build tensorflow graph with input and two output layers

        Parameters
        -----------
        x_tf: tf.tensors (n_observations, n_channels)
            placeholder of recording for running tensorflow

        channel_index: np.array (n_channels, n_neigh)
            Each row indexes its neighboring channels. For example,
            channel_index[c] is the index of neighboring channels (including
            itself) If any value is equal to n_channels, it is nothing but
            a placeholder in a case that a channel has less than n_neigh
            neighboring channels

        threshold: int
            threshold on a probability to determine
            location of spikes

        Returns
        -------
        spike_index_tf: tf tensor (n_spikes, 2)
            tensorflow tensor that produces spike_index
        """
        ######################
        # Loading parameters #
        ######################

        # TODO: need to ask why we are sending different channel indexes
        # save neighbor channel index
        small_channel_index = channel_index[:, :n_neigh]

        # placeholder for input recording
        x_tf = tf.placeholder("float", [None, None])

        # Temporal shape of input
        T = tf.shape(x_tf)[0]

        ####################
        # Building network #
        ####################

        # input tensor into CNN - add one dimension at the beginning and
        # at the end
        x_cnn_tf = tf.expand_dims(tf.expand_dims(x_tf, -1), 0)

        vars_dict, layer11 = cls._make_network(x_cnn_tf,
                                               waveform_length,
                                               filters_size,
                                               n_neigh,
                                               padding='SAME')
        W2 = vars_dict['W2']
        b2 = vars_dict['b2']

        K1, K2 = filters_size

        # first spatial layer
        zero_added_layer11 = tf.concat((tf.transpose(layer11, [2, 0, 1, 3]),
                                        tf.zeros((1, 1, T, K2))),
                                       axis=0)

        temp = tf.transpose(tf.gather(zero_added_layer11, small_channel_index),
                            [0, 2, 3, 1, 4])

        temp2 = conv2d_VALID(tf.reshape(temp, [-1, T, n_neigh, K2]), W2) + b2

        o_layer = tf.transpose(temp2, [2, 1, 0, 3])

        ################################
        # Output layer transformations #
        ################################

        o_layer_val = tf.squeeze(o_layer)

        # probability output - just sigmoid of output layer
        probability_tf = tf.sigmoid(o_layer_val)

        # spike index output (local maximum crossing a threshold)
        temporal_max = tf.squeeze(max_pool(o_layer, [1, 3, 1, 1]) - 1e-8)

        higher_than_max_pool = o_layer_val >= temporal_max

        higher_than_threshold = (o_layer_val >
                                 np.log(threshold / (1 - threshold)))

        both_higher = tf.logical_and(higher_than_max_pool,
                                     higher_than_threshold)

        index_all = tf.cast(tf.where(both_higher), 'int32')

        spike_index_tf = cls._remove_edge_spikes(x_tf, index_all,
                                                 waveform_length)

        # waveform output from spike index output
        waveform_tf = cls._make_waveform_tf(x_tf, spike_index_tf,
                                            channel_index, waveform_length)

        return x_tf, spike_index_tf, probability_tf, waveform_tf, vars_dict

    @classmethod
    def _remove_edge_spikes(cls, x_tf, spike_index_tf, waveform_length):
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
        max_spike_time = tf.shape(x_tf)[0] - R

        idx_middle = tf.logical_and(spike_index_tf[:, 0] > min_spike_time,
                                    spike_index_tf[:, 0] < max_spike_time)

        return tf.boolean_mask(spike_index_tf, idx_middle)

    @classmethod
    def _make_waveform_tf(cls, x_tf, spike_index_tf, channel_index, wf_length):
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

        wf_length: int
            temporal length of waveform

        Returns
        -------
        tf tensor (n_spikes, wf_length, n_neigh)
        """
        R = int((wf_length-1)/2)  # half waveform length
        T = tf.shape(x_tf)[0]  # length of recording

        # get waveform temporally

        # make indexes with the appropriate waveform length, centered at zero
        # shape: [1, wf_length]
        waveform_indexes = tf.expand_dims(tf.range(-R, R+1), 0)
        # get all spike times, shape: [n_spikes, 1]
        spike_times = tf.expand_dims(spike_index_tf[:, 0], -1)
        # shift indexes and add two dimensions, shape: [n_spikes, wf_length]
        _ = tf.add(spike_times, waveform_indexes)
        # add two trailing extra dimensions, shape: [n_spikes, wf_length, 1, 1]
        wf_temporal = tf.expand_dims(tf.expand_dims(_, -1), -1)

        # get waveform spatially
        # get neighbors for main channels in the spike index
        # shape: [n_spikes, n_neigh]
        _ = tf.gather(channel_index, spike_index_tf[:, 1])
        # add one dimension to the left and one to the right
        # shape: [n_spikes, 1, n_neigh, 1]
        wf_spatial = tf.expand_dims(tf.expand_dims(_, 1), -1)

        # build spatio-temporal index

        # tile temporal indexes on the number of channels and spatial indexes
        # on the waveform length, then concatenate
        # FIXME: there is a mismatch here, we aren't we using self.n_neigh
        n_neigh = channel_index.shape[1]
        _ = (tf.tile(wf_temporal, (1, 1, n_neigh, 1)),
             tf.tile(wf_spatial, (1, wf_length, 1, 1)))
        idx = tf.concat(_, 3)

        # add one extra value in the channels dimension
        x_tf_zero_added = tf.concat([x_tf, tf.zeros((T, 1))], axis=1)

        return tf.gather_nd(x_tf_zero_added, idx)

    def restore(self, sess):
        """Restore tensor values
        """
        self.saver.restore(sess, self.path_to_model)

    def predict(self, recordings, output_names=('spike_index',)):
        """Make predictions

        Parameters
        ----------
        output: tuple
            Which output layers to return, valid options are: spike_index,
            waveform and probability


        Returns
        -------
        tuple
            A tuple of numpy.ndarrays, one for every element in output_names
        """
        output_tensors = [getattr(self, name+'_tf') for name in output_names]

        with tf.Session() as sess:
            self.restore(sess)

            output = sess.run(output_tensors,
                              feed_dict={self.x_tf: recordings})

        return output

    def fit(self, x_train, y_train):
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
        ######################
        # Loading parameters #
        ######################

        logger = logging.getLogger(__name__)

        # get parameters
        n_data, waveform_length_train, n_neighbors_train = x_train.shape

        if self.waveform_length != waveform_length_train:
            raise ValueError('waveform length from network ({}) does not '
                             'match training data ({})'
                             .format(self.waveform_length,
                                     waveform_length_train))

        if self.n_neighbors != n_neighbors_train:
            raise ValueError('number of n_neighbors from network ({}) does '
                             'not match training data ({})'
                             .format(self.n_neigh,
                                     n_neighbors_train))

        ####################
        # Building network #
        ####################

        # x and y input tensors
        x_tf = tf.placeholder("float", [self.n_batch, self.waveform_length,
                                        self.n_neighbors])
        y_tf = tf.placeholder("float", [self.n_batch])

        input_tf = tf.expand_dims(x_tf, -1)

        vars_dict, layer11 = (NeuralNetDetector
                              ._make_network(input_tf,
                                             self.waveform_length,
                                             self.filters_size,
                                             self.n_neighbors,
                                             padding='VALID'))

        W2 = vars_dict['W2']
        b2 = vars_dict['b2']

        # third layer: spatial convolution
        o_layer = tf.squeeze(conv2d_VALID(layer11, W2) + b2)

        ##########################
        # Optimization objective #
        ##########################

        # cross entropy
        _ = tf.nn.sigmoid_cross_entropy_with_logits(logits=o_layer,
                                                    labels=y_tf)
        cross_entropy = tf.reduce_mean(_)

        weights = tf.trainable_variables()

        # regularization term
        l2_regularizer = (tf.contrib.layers
                          .l2_regularizer(scale=self.l2_reg_scale))

        regularization = tf.contrib.layers.apply_regularization(l2_regularizer,
                                                                weights)

        regularized_loss = cross_entropy + regularization

        # train step
        train_step = (tf.train.AdamOptimizer(self.train_step_size)
                        .minimize(regularized_loss))

        ############
        # Training #
        ############

        # saver
        saver = tf.train.Saver(vars_dict)

        logger.debug('Training detector network...')

        bar = tqdm(total=self.n_iter)

        with tf.Session() as sess:

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for i in range(0, self.n_iter):

                # sample n_batch observations from 0, ..., n_data
                idx_batch = np.random.choice(n_data, self.n_batch,
                                             replace=False)

                res = sess.run(
                    [train_step, regularized_loss],
                    feed_dict={
                        x_tf: x_train[idx_batch],
                        y_tf: y_train[idx_batch]
                    })

                bar.update(i + 1)

                # if not i % 100:
                #    logger.debug('Loss: %s', res[1])

            logger.debug('Saving network: %s', self.path_to_model)
            saver.save(sess, self.path_to_model)

            # estimate tp and fp with a sample
            idx_batch = np.random.choice(n_data, self.n_batch, replace=False)

            output = sess.run(o_layer, feed_dict={x_tf: x_train[idx_batch]})
            y_test = y_train[idx_batch]

            tp = np.mean(output[y_test == 1] > 0)
            fp = np.mean(output[y_test == 0] > 0)

            logger.debug('Approximate training true positive rate: '
                         + str(tp) + ', false positive rate: ' + str(fp))
        bar.close()

        path_to_params = change_extension(self.path_to_model, 'yaml')
        logger.debug('Saving network parameters: %s', path_to_params)
        save_detect_network_params(filters_size=self.filters_size,
                                   waveform_length=self.waveform_length,
                                   n_neighbors=self.n_neighbors,
                                   output_path=path_to_params)
