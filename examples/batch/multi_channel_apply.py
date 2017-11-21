"""
Filter data using BatchProcessor.multi_channel_apply
"""

import logging
import os

import matplotlib.pyplot as plt

from yass.batch.new import BatchProcessor
from yass.batch import RecordingsReader
from yass.preprocess.filter import butterworth


logging.basicConfig(level=logging.INFO)


path_to_neuropixel_data = (os.path.expanduser('~/data/ucl-neuropixel'
                           '/rawDataSample.bin'))
path_to_filtered_data = (os.path.expanduser('~/data/ucl-neuropixel'
                         '/tmp/filtered_multi.bin'))

# create batch processor for the data
bp = BatchProcessor(path_to_neuropixel_data,
                    dtype='int16', n_channels=385, data_format='wide',
                    max_memory='500MB')

# appply a multi channel transformation, each batch will be a temporal
# subset with observations from all selected n_channels, the size
# of the subset is calculated depending on max_memory
bp.multi_channel_apply(butterworth, path_to_filtered_data,
                       channels='all',
                       low_freq=300, high_factor=0.1,
                       order=3, sampling_freq=30000)

# let's visualize the results
raw = RecordingsReader(path_to_neuropixel_data, dtype='int16',
                       n_channels=385, data_format='wide')

# you do not need to specify the format since multi_channel_apply
# saves a yaml file with such parameters
filtered = RecordingsReader(path_to_filtered_data)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(raw[:2000, 0])
ax2.plot(filtered[:2000, 0])
plt.show()
