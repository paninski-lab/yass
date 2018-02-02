import tensorflow as tf

from . import NeuralNetDetector


# TODO: what is this for?
def get_score(score_train_tf, spike_index_tf, T, n_features, c_idx):
    score_train_tf_zero = tf.concat((tf.transpose(score_train_tf, [1, 0, 2]),
                                     tf.zeros((1, T, n_features))
                                     ), axis=0)
    temp = tf.transpose(tf.gather(score_train_tf_zero, c_idx), [2, 0, 3, 1])
    score = tf.gather_nd(temp, spike_index_tf)

    return score


def load_rotation(detector_filename, autoencoder_filename):
    """
    Load neural network rotation matrix
    """

    # FIXME: this function should not ask for detector_filename, it is not
    # needed
    nnd = NeuralNetDetector(detector_filename, autoencoder_filename)

    with tf.Session() as sess:
        nnd.saver_ae.restore(sess, nnd.path_to_ae_model)
        rotation = sess.run(nnd.W_ae)

    return rotation
