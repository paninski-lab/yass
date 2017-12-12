"""
Preprocessing example, filtering and standarizing a single channel
in neuropixel data.

This example shows the usage of seveal preprocessing functions included in
yass you most likely will not use these functions directly but use the
pre-built pipeline which uses these functions under the hood and implements
batch processing logic on top of them
"""

# TODO: maybe split this into different files and add a reference
# in their corresponding functions

import os
import numpy as np
import matplotlib.pyplot as plt

from yass import preprocess
from yass import geometry
from yass.preprocess import pca
from yass.preprocess import whiten

# path to neuropixel data
root = os.path.expanduser('~/data/ucl-neuropixel')
path_to_data = os.path.join(root, 'rawDataSample.bin')
path_to_geom = os.path.join(root, 'channel_positions.npy')

observations = 1800000
n_channels = 385
sampling_freq = 30000


# path to 49 channels data
root = os.path.expanduser('~/data/yass')
path_to_data = os.path.join(root, 'ej49_data1_set1.bin')
path_to_geom = os.path.join(root, 'ej49_geometry1.txt')

observations = 6000000
n_channels = 49
sampling_freq = 20000


rec = np.fromfile(path_to_data, dtype='int16').reshape(observations,
                                                       n_channels)
rec.shape

# TODO: check number of channels in neuropixel data
geom = geometry.parse(path_to_geom, n_channels)
neighbors = geometry.find_channel_neighbors(geom, radius=70)

# get some observations from channel 0
raw_data = rec[50000:51000, 0]


filtered = preprocess.butterworth(raw_data,
                                  low_freq=300,
                                  high_factor=0.1,
                                  order=3,
                                  sampling_freq=sampling_freq)

standarized = preprocess.standarize(filtered, sampling_freq=sampling_freq)


fix, (ax1, ax2, ax3) = plt.subplots(nrows=3)
ax1.plot(raw_data)
ax1.set_title('Raw')
ax2.plot(filtered)
ax2.set_title('Filtered')
ax3.plot(standarized)
ax3.set_title('Standarized')
plt.tight_layout()
plt.show()

spike_size = 40
whitened = whiten.apply(rec, neighbors, spike_size)

plt.plot(whitened[:1000, 0])
plt.show()

# run threshold detection, not sure if this is the right
# place for the threshold detector to be
standarized = standarized.reshape(1000, 1)
spike_index = preprocess.detect.threshold(rec, neighbors, spike_size,
                                          std_factor=1)

n_features = 3

suff_stat, spikes_per_channel = pca.suff_stat(rec, spike_index,
                                              spike_size)

proj = pca.project(suff_stat, spikes_per_channel, n_features, neighbors)


scores = pca.score(spike_index, rot, neighbors, geom, batch_size, BUFF, nBatches,
                    wf_path, scale_to_save)
