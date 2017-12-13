import tensorflow as tf


def get_score(score_train_tf, spike_index_tf, T, n_features, c_idx):
    score_train_tf_zero = tf.concat((tf.transpose(score_train_tf, [1, 0, 2]),
                                     tf.zeros((1, T, n_features))
                                     ), axis=0)
    temp = tf.transpose(tf.gather(score_train_tf_zero, c_idx), [2, 0, 3, 1])
    score = tf.gather_nd(temp, spike_index_tf)

    return score
