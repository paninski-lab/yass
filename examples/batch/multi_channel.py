import os

from yass.batch import BatchProcessor


path_to_neuropixel_data = (os.path.expanduser('~/data/ucl-neuropixel'
                           '/rawDataSample.bin'))


bp = BatchProcessor(path_to_neuropixel_data,
                    dtype='int16', n_channels=385, data_format='wide',
                    max_memory='1MB')

# now, let's to some multi_channel operations, here we will traverse all
# channels and all observations, each batch will contain a subset in the
# temporal dimension, the window size is determined by max_memory
data = bp.multi_channel()

for d, _, idx in data:
    print('Shape: {}. Index: {}'.format(d.shape, idx))

# we can specify the temporal limits and subset channels
data = bp.multi_channel(from_time=100000, to_time=200000, channels=[0, 1, 2])

for d, _, idx in data:
    print('Shape: {}. Index: {}'.format(d.shape, idx))


# we can also create a BatchProcessor with a buffer
bp2 = BatchProcessor(path_to_neuropixel_data,
                     dtype='int16', n_channels=385, data_format='wide',
                     max_memory='1KB', buffer_size=10)

data = bp2.multi_channel(from_time=0, to_time=100000, channels=[0, 1, 2])

for d, idx_local, idx in data:
    print('Shape: {}. Index (local): {}. Index: {}'.format(d.shape, idx_local,
                                                           idx))
