import numpy as np
import tensorflow as tf
import progressbar

from .utils import weight_variable, bias_variable, conv2d, conv2d_VALID


def train_triage(x_train, y_train, n_filters, n_iter, n_batch, l2_reg_scale,
                 train_step_size, nn_name):
    """
        Trains the triage network

        Parameters:
        -----------
        x_train: np.array
            [number of data, temporal length, number of channels] training data
            for the triage network.
        y_train: np.array
            [number of data] training label for the triage network.
        nn_name: string
            name of the .ckpt to be saved.
    """
    # get parameters
    ndata, R, C = x_train.shape
    K1, K2 = n_filters

    # x and y input tensors
    x_tf = tf.placeholder("float", [n_batch, R, C])
    y_tf = tf.placeholder("float", [n_batch])

    # first layer: temporal feature
    W1 = weight_variable([R, 1, 1, K1])
    b1 = bias_variable([K1])
    layer1 = tf.nn.relu(conv2d_VALID(
        tf.expand_dims(x_tf, -1), W1) + b1)

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
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=o_layer, labels=y_tf))

    # regularization term
    weights = tf.trainable_variables()
    l2_regularizer = tf.contrib.layers.l2_regularizer(
        scale=l2_reg_scale)
    regularization_penalty = tf.contrib.layers.apply_regularization(
        l2_regularizer, weights)
    regularized_loss = cross_entropy + regularization_penalty

    # train step
    train_step = tf.train.AdamOptimizer(
        train_step_size).minimize(regularized_loss)

    # saver
    saver = tf.train.Saver(
        {"W1": W1, "W11": W11, "W2": W2, "b1": b1, "b11": b11, "b2": b2})

    ############
    # training #
    ############

    bar = progressbar.ProgressBar(maxval=n_iter)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(0, n_iter):
            idx_batch = np.random.choice(ndata, n_batch, replace=False)
            sess.run(train_step, feed_dict={x_tf: x_train[
                     idx_batch], y_tf: y_train[idx_batch]})
            bar.update(i+1)
        saver.save(sess, nn_name)

        idx_batch = np.random.choice(ndata, n_batch, replace=False)
        output = sess.run(o_layer, feed_dict={x_tf: x_train[idx_batch]})
        y_test = y_train[idx_batch]
        tp = np.mean(output[y_test == 1] > 0)
        fp = np.mean(output[y_test == 0] > 0)

        print('Approximate training true positive rate: ' +
              str(tp)+', false positive rate: '+str(fp))
    bar.finish()
