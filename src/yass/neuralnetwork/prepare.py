import tensorflow as tf
import numpy as np

from yass.geometry import order_channels_by_distance
from yass.neuralnetwork import NeuralNetDetector, NeuralNetTriage

def prepare_nn(channel_index, whiten_filter,
                        threshold_detect, threshold_triage,
                        detector_filename, 
                        autoencoder_filename, 
                        triage_filename):
    
    # placeholder for input recording
    x_tf = tf.placeholder("float", [None, None])
    
    # load Neural Net's
    NND = NeuralNetDetector(detector_filename, autoencoder_filename)
    NNT = NeuralNetTriage(triage_filename)
    
    # check if number of neighboring channel in nn matches with 
    # channel index
    if channel_index.shape[1] != NND.filters_dict['n_neighbors']:
        raise ValueError('Number of neighboring channels from neighbors is {}'
                         'but the trained Neural Net expects {} neighbors, they must match'
                     .format(nneigh, NND.filters_dict['n_neighbors']))

    # make spike_index tensorflow tensor
    spike_index_tf = NND.make_detection_tf_tensors(x_tf,
                                                channel_index,
                                                threshold_detect)
    
    # make waveform tensorflow tensor
    waveform_tf = make_waveform_tf_tensor(x_tf, 
                                          spike_index_tf, 
                                          channel_index, 
                                          NND.filters_dict['size'])
    
    # make score tensorflow tensor from waveform
    score_tf = NND.make_score_tf_tensor(waveform_tf)
    
    # remove uncentered spike index
    # if the energy in the main channel is less than 
    # neighoring channels, it is uncentered
    (score_keep_tf, wf_keep_tf, 
     spike_index_keep_tf) = remove_uncetered(score_tf, 
                                             waveform_tf, 
                                             spike_index_tf)
    

    # run neural net triage
    idx_clean = NNT.triage_wf(wf_keep_tf, threshold_triage)
    score_clear_tf = tf.boolean_mask(score_keep_tf, idx_clean)
    spike_index_clear_tf = tf.boolean_mask(spike_index_keep_tf, idx_clean)

    # whiten score
    whiten_score_clear_tf = make_whitened_score(score_clear_tf,
                                                spike_index_clear_tf[:, 1],
                                                whiten_filter)
    
    # gather all output tensors
    output_tf = (whiten_score_clear_tf, spike_index_clear_tf, spike_index_tf)
    
    return x_tf, output_tf, NND, NNT

def make_whitened_score(score_tf, channel_index, whiten_filter):
    """?

    Parameters
    ----------
    ?

    Returns
    -------
    ?
    """
    
    whiten_filter_per_data = tf.gather(tf.transpose(
        whiten_filter, [2, 0, 1]), channel_index)
    
    return tf.matmul(score_tf, whiten_filter_per_data)


def remove_uncetered(score_tf, waveform_tf, spike_index_tf):
    
    energy_tf = tf.reduce_sum(tf.square(score_tf), 1)
    idx_centered = tf.equal(tf.argmax(energy_tf, 1), 0)
    
    wf_keep_tf = tf.boolean_mask(waveform_tf, idx_centered)
    score_keep_tf = tf.boolean_mask(score_tf, idx_centered)
    spike_index_keep_tf = tf.boolean_mask(spike_index_tf, idx_centered)

    return score_keep_tf, wf_keep_tf, spike_index_keep_tf
    
    
def make_waveform_tf_tensor(x_tf, spike_index_tf, channel_index, waveform_length):
    
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
    
    wf_idx = tf.concat((tf.tile(wf_temporal_expand,(1, 1, nneigh, 1)),
                        tf.tile(wf_spatial_expand,(1, waveform_length, 1, 1))
                       ),3)
    
    return tf.gather_nd(x_tf,wf_idx)