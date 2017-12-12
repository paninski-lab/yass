"""
Preprocessing example, filtering and standarizing a single channel
in neuropixel data.

This example shows the usage of seveal preprocessing functions included in
yass you most likely will not use these functions directly but use the
pre-built pipeline which uses these functions under the hood and implements
batch processing logic on top of them
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from yass import preprocess
from yass import geometry

# path to neuropixel data
root = os.path.expanduser('~/data/ucl-neuropixel')
path_to_data = os.path.join(root, 'rawDataSample.bin')
path_to_geom = os.path.join(root, 'channel_positions.npy')

observations = 1800000
n_channels = 385
sampling_freq = 30000

rec = np.fromfile(path_to_data, dtype='int16').reshape(observations,
                                                       n_channels)
rec.shape

# TODO: check number of channels
# geom = geometry.parse(path_to_geom, n_channels)
# neighbors = geometry.find_channel_neighbors(geom, radius=70)

# get some observations from channel 0
raw_data = rec[50000:51000, 0]


filtered = preprocess.butterworth(raw_data,
                                  low_freq=300,
                                  high_factor=0.1,
                                  order=3,
                                  sampling_freq=sampling_freq)

standarized = preprocess.standarize(filtered, sampling_freq=sampling_freq)

# TODO: add whitening example


fix, (ax1, ax2, ax3) = plt.subplots(nrows=3)
ax1.plot(raw_data)
ax1.set_title('Raw')
ax2.plot(filtered)
ax2.set_title('Filtered')
ax3.plot(standarized)
ax3.set_title('Standarized')
plt.tight_layout()
plt.show()
