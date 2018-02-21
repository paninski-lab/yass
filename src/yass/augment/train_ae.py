import numpy as np
import tensorflow as tf
import progressbar


def train_ae(x_train, y_train, n_feature, n_iter, n_batch, train_step_size,
             nn_name):
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
        nn_name: string
            name of the .ckpt to be saved.
    """

    # parameters
    n_data, n_input = x_train.shape

    # input tensors
    x_ = tf.placeholder("float", [n_batch, n_input])
    y_ = tf.placeholder("float", [n_batch, n_input])

    # encoding
    W_ae = tf.Variable(tf.random_uniform(
        (n_input, n_feature), -1.0 / np.sqrt(n_input), 1.0 / np.sqrt(n_input)))
    h = tf.matmul(x_, W_ae)

    # decoding
    Wo = tf.transpose(W_ae)
    y_tf = tf.matmul(h, Wo)

    # training
    meansq = tf.reduce_mean(tf.square(y_-y_tf))
    train_step = tf.train.GradientDescentOptimizer(
        train_step_size).minimize(meansq)

    # saver
    saver_ae = tf.train.Saver({"W_ae": W_ae})

    ############
    # training #
    ############

    bar = progressbar.ProgressBar(maxval=n_iter)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(0, n_iter):
            idx_batch = np.random.choice(n_data, n_batch, replace=False)
            sess.run(train_step, feed_dict={x_: x_train[
                     idx_batch], y_: y_train[idx_batch]})
            bar.update(i+1)
        saver_ae.save(sess, nn_name)
    bar.finish()
