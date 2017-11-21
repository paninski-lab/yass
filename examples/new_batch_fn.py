"""
%load_ext autoreload
%autoreload 2
"""

import logging
import os

import numpy as np

from yass.batch.new import BatchProcessor
from yass.batch import RecordingsReader

logging.basicConfig(level=logging.DEBUG)


root = '/Users/Edu/data/yass-benchmarks'
# root = '/ssd/data/eduardo/yass-benchmarks'

path_to_long = os.path.join(root, 'long.bin')
path_to_wide = os.path.join(root, 'wide.bin')
path_to_out = os.path.join(root, 'out.bin')


wide_shape = (50, 2000000)
long_shape = (2000000, 50)

big_long = np.memmap(path_to_long, 'int64', 'w+', shape=long_shape)
big_wide = np.memmap(path_to_wide, 'int64', 'w+', shape=wide_shape)

big_long.shape
big_wide.shape

# slow
big_long[:, 1]
x_long = big_long[:, 1] + 1
big_long[:, 1] = x_long

# fast
big_wide[0, :]
x_wide = big_wide[0, :] + 1
big_wide[0, :] = x_wide


def dummy(arr):
    return arr + 1


x_long = dummy(big_long[(slice(0, 2000000, None), 1)])
big_long[(slice(0, 2000000, None), 1)] = x_long
big_long.flush()

x_long = dummy(big_long[:, 1])
big_long[:, 1] = x_long


bp_long = BatchProcessor(path_to_long,
                         dtype='int64', n_channels=50, data_format='long',
                         max_memory='500MB')

path = bp_long.single_channel_apply(dummy, path_to_out)

out = RecordingsReader(path)
out


bp_wide = BatchProcessor(path_to_wide,
                         dtype='int64', n_channels=50, data_format='wide',
                         max_memory='500MB')

path = bp_wide.single_channel_apply(dummy, path_to_out)
out = RecordingsReader(path)
out


path = bp_long.multi_channel_apply(dummy, path_to_out)
out = RecordingsReader(path)
out

path = bp_wide.multi_channel_apply(dummy, path_to_out)
out = RecordingsReader(path)
out
