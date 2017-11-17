"""
%load_ext autoreload
%autoreload 2
"""

import logging
import os

import numpy as np

from yass.batch.new import BatchProcessor

logging.basicConfig(level=logging.INFO)


root = '/Users/Edu/data/yass-benchmarks'
path_to_wide_data = os.path.join(root, 'sample_wide.bin')
path_to_long_data = os.path.join(root, 'sample_long.bin')
path_to_wide_out = os.path.join(root, 'out_wide.bin')
path_to_long_out = os.path.join(root, 'out_long.bin')

path_to_big = os.path.join(root, 'big.bin')
path_to_big_out = os.path.join(root, 'out_big.bin')
np.memmap(path_to_big, 'int64', 'w+', shape=(5000000, 500))

wide_shape = (500, 2000000)
long_shape = (2000000, 500)
ones = np.ones(wide_shape)
series = np.array(range(wide_shape[1]))
data_wide = (series * ones).astype('int64')
data_long = data_wide.T

data_wide
data_long


data_wide.tofile(path_to_wide_data)
data_long.tofile(path_to_long_data)


bp_wide = BatchProcessor(path_to_wide_data,
                         dtype='int64', channels=50, data_format='wide',
                         max_memory='3GB')

bp_long = BatchProcessor(path_to_long_data,
                         dtype='int64', channels=50, data_format='long',
                         max_memory='3GB')


def dummy(arr):
    return arr + 1


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


bp = BatchProcessor(path_to_big,
                    dtype='int64', channels=500, data_format='long',
                    max_memory='3GB')

dtype, path = bp.single_channel_apply(dummy, path_to_big_out, 'int64')
loaded = np.fromfile(path, dtype).reshape(long_shape)
loaded
loaded.shape

dtype, path = bp.multi_channel_apply(dummy, path_to_big_out, 'int64')
loaded = np.fromfile(path, dtype).reshape(long_shape)
loaded
loaded.shape
