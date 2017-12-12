"""
Preprocessing example, filtering and standarizing a single channel
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from yass import preprocess
from yass import geometry

root = '/Users/Edu/data/yass'

path_to_data = os.path.join(root, 'ej49_data1_set1.bin')
path_to_geom = os.path.join(root, 'ej49_geometry1.txt')

n_channels = 49
sampling_freq = 20000

rec = np.fromfile(path_to_data, dtype='int16').reshape(6000000, n_channels)
rec.shape

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
ax2.plot(filtered)
ax3.plot(standarized)
plt.tight_layout()
plt.show()
