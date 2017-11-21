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

path_to_long = os.path.join(root, 'long.bin')
path_to_out = os.path.join(root, 'out.bin')


def dummy(arr):
    return arr + 1


bp = BatchProcessor(path_to_long,
                    dtype='int64', channels=500, data_format='long',
                    max_memory='100MB')

dtype, path = bp.single_channel_apply(dummy, path_to_out, 'int64')
