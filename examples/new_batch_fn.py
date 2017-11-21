"""
%load_ext autoreload
%autoreload 2
"""

import logging
import os

import numpy as np

from yass.batch.new import BatchProcessor

logging.basicConfig(level=logging.INFO)


# root = '/Users/Edu/data/yass-benchmarks'
root = '/ssd/data/eduardo/yass-benchmarks'

path_to_long = os.path.join(root, 'long.bin')
path_to_wide = os.path.join(root, 'wide.bin')
path_to_out = os.path.join(root, 'out.bin')


wide_shape = (500, 2000000)
long_shape = (2000000, 500)

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

x_long = dummy(big_long[:, 1])
big_long[:, 1] = x_long



bp_long = BatchProcessor(path_to_long,
                         dtype='int64', channels=500, data_format='long',
                         max_memory='500MB')

dtype, path = bp_long.single_channel_apply(dummy, path_to_out, 'int64',
                                           channels=1)
loaded = np.fromfile(path, dtype).reshape(long_shape)


bp_wide = BatchProcessor(path_to_wide,
                         dtype='int64', channels=500, data_format='long',
                         max_memory='500MB')

dtype, path = bp_wide.single_channel_apply(dummy, path_to_out, 'int64',
                                           channels=1)
loaded = np.fromfile(path, dtype).reshape(long_shape)


dtype, path = bp.multi_channel_apply(dummy, path_to_out, 'int64')
loaded = np.fromfile(path, dtype).reshape(long_shape)
loaded
loaded.shape


ones = np.ones(wide_shape)
series = np.array(range(wide_shape[1]))
data_wide = (series * ones).astype('int64')
data_long = data_wide.T



path_to_wide_data = os.path.join(root, 'sample_wide.bin')
path_to_long_data = os.path.join(root, 'sample_long.bin')
path_to_wide_out = os.path.join(root, 'out_wide.bin')
path_to_long_out = os.path.join(root, 'out_long.bin')


data_wide.tofile(path_to_wide_data)
data_long.tofile(path_to_long_data)


bp_wide = BatchProcessor(path_to_wide_data,
                         dtype='int64', channels=50, data_format='wide',
                         max_memory='3GB')

bp_long = BatchProcessor(path_to_long_data,
                         dtype='int64', channels=50, data_format='long',
                         max_memory='3GB')


data_wide
data_wide.shape

dtype, path = bp_wide.single_channel_apply(dummy, path_to_wide_out, 'int64')
loaded = np.fromfile(path, dtype).reshape(wide_shape)
loaded
loaded.shape

dtype, path = bp_wide.multi_channel_apply(dummy, path_to_wide_out, 'int64')
loaded = np.fromfile(path, dtype).reshape(wide_shape)
loaded
loaded.shape


data_long
data_long.shape

dtype, path = bp_long.single_channel_apply(dummy, path_to_long_out, 'int64')
loaded = np.fromfile(path, dtype).reshape(long_shape)
loaded
loaded.shape

dtype, path = bp_long.multi_channel_apply(dummy, path_to_long_out, 'int64')
loaded = np.fromfile(path, dtype).reshape(long_shape)
loaded
loaded.shape
