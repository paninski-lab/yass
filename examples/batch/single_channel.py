import os

from yass.batch import BatchProcessor


path_to_neuropixel_data = (os.path.expanduser('~/data/ucl-neuropixel'
                           '/rawDataSample.bin'))


bp = BatchProcessor(path_to_neuropixel_data,
                    dtype='int16', n_channels=385, data_format='wide',
                    max_memory='1MB')

# there are two ways of traversing the data: single_channel and multi_channel
# single_channel means that the data in a single batch comes from only one
# channel, multi_channel means that a batch can contain data from multiple
# channels, let's take a look at single_channel operations

# traverse the whole dataset, one channel at a time
data = bp.single_channel()

# this will raise an error since we cannot fit all observations for a single
# channel in memory, so we either increase max_memory or set
# force_complete_channel_batch to False
for d in data:
    print(d.shape)

# When force_complete_channel_batch is False, each batch does not necessarily
# correspond to all observations in the channel, the channel can be splitted
# in several batches (although every batch data is guaranteed to come from
# a single channel), in this case, every channel is splitted in two parts
data = bp.single_channel(force_complete_channel_batch=False)

for d, i in data:
    print(d.shape, 'Data from channel {}'.format(i))


# finally, we can traverse a single channel in a temporal subset
data = bp.single_channel(from_time=100000, to_time=200000)

for d in data:
    print(d.shape)

# we can select specific channels
data = bp.single_channel(from_time=100000, to_time=200000, channels=[0, 1, 2])

for d in data:
    print(d.shape)


# now, let's to some multi_channel operations, here we will traverse all
# channels and all observations, each batch will contain a subset in the
# temporal dimension, the window size is determined by max_memory
data = bp.multi_channel()

for d in data:
    print(d.shape)

# we can specify the temporal limits and subset channels
data = bp.multi_channel(from_time=100000, to_time=200000, channels=[0, 1, 2])

for d in data:
    print(d.shape)
