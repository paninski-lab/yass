"""
Functions for threshold detection
"""
import os
import logging

import numpy as np
from scipy.signal import argrelmin

def voltage_threshold(recording, threshold, order=5):

    T, C = recording.shape
    spike_index = np.zeros((0, 2), 'int32')
    energy = np.zeros(0, 'float32')
    for c in range(C):
        single_chan_rec = recording[:, c]
        index = argrelmin(single_chan_rec, order=order)[0]
        index = index[single_chan_rec[index] < -threshold]
        spike_index_temp = np.vstack((index,
                                      np.ones(len(index), 'int32')*c)).T
        spike_index = np.concatenate((spike_index, spike_index_temp), axis=0)
        energy_ = np.abs(single_chan_rec[index])
        energy = np.hstack((energy, energy_))

    return spike_index, energy