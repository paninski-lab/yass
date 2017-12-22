"""
Filter data using BatchProcessor.single_channel_apply
"""

import logging
import os

import matplotlib.pyplot as plt

from yass.batch import BatchProcessor
from yass.batch import RecordingsReader
from yass.preprocess.filter import butterworth_single_channel


logging.basicConfig(level=logging.INFO)


path_to_neuropixel_data = (os.path.expanduser('~/data/ucl-neuropixel'
                           '/rawDataSample.bin'))
path_to_filtered_data = (os.path.expanduser('~/data/ucl-neuropixel'
                         '/tmp/filtered.bin'))

# create batch processor for the data
bp = BatchProcessor(path_to_neuropixel_data,
                    dtype='int16', n_channels=385, data_format='wide',
                    max_memory='500MB')

# appply a single channel transformation, each batch will be all observations
# from one channel
bp.single_channel_apply(butterworth_single_channel, path_to_filtered_data,
                        low_freq=300, high_factor=0.1,
                        order=3, sampling_freq=30000)

# let's visualize the results
raw = RecordingsReader(path_to_neuropixel_data, dtype='int16',
                       n_channels=385, data_format='wide')

# you do not need to specify the format since single_channel_apply
# saves a yaml file with such parameters
filtered = RecordingsReader(path_to_filtered_data)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(raw[:2000, 0])
ax2.plot(filtered[:2000, 0])
plt.show()
