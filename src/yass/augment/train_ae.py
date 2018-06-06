import tensorflow as tf
from sklearn.decomposition import PCA


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

    pca = PCA(n_components=n_feature).fit(x_train)

    # encoding
    W_ae = tf.Variable((pca.components_.T).astype('float32'))

    # saver
    saver_ae = tf.train.Saver({"W_ae": W_ae})

    ############
    # training #
    ############

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver_ae.save(sess, nn_name)
