import tensorflow as tf

from .utils import max_pool


def remove_duplicate_spikes_by_energy(energy_train_tf, T, c_idx,
                                      temporal_window):

    temporal_max_energy = max_pool(
        tf.expand_dims(tf.expand_dims(energy_train_tf, 0), -1),
        [1, temporal_window, 1, 1])
    zero_added_output = tf.concat(
        (temporal_max_energy, tf.zeros((1, T, 1, 1))), axis=2)
    max_neigh_energy = tf.transpose(tf.squeeze(tf.reduce_max(
        tf.gather(tf.transpose(zero_added_output, [2, 1, 0, 3]), c_idx),
        axis=1)))
    return tf.where(tf.logical_and(energy_train_tf > 0,
                                   energy_train_tf >= max_neigh_energy))
