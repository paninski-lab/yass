import logging
import os

import matplotlib.pyplot as plt

from yass.batch.new import BatchProcessor
from yass.batch import RecordingsReader
from yass.preprocess.filter import butterworth_single_channel

logging.basicConfig(level=logging.INFO)


path_to_neuropixel_data = (os.path.expanduser('~/data/ucl-neuropixel'
                           '/rawDataSample.bin'))
path_to_filtered_data = (os.path.expanduser('~/data/ucl-neuropixel'
                         '/tmp/filtered.bin'))


bp = BatchProcessor(path_to_neuropixel_data,
                    dtype='int16', n_channels=385, data_format='wide',
                    max_memory='500MB')


bp.single_channel_apply(butterworth_single_channel, path_to_filtered_data,
                        low_freq=300, high_factor=0.1,
                        order=3, sampling_freq=30000)

# let's visualize the results
raw = RecordingsReader(path_to_neuropixel_data, dtype='int16',
                       n_channels=385, data_format='wide')
filtered = RecordingsReader(path_to_filtered_data)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(raw[:2000, 0])
ax2.plot(filtered[:2000, 0])
plt.show()
