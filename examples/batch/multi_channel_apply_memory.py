# coding: utf-8

# See notebook:
# https://github.com/paninski-lab/yass-examples/blob/master/batch/multi_channel_apply_memory.ipynb

"""
Applying transformations to large files in batches:

BatchProcessor.multi_channel_apply lets you apply transformations to
batches of data where every batch has observations from every channel.

This example show how to extract information from a large file by
processing it in batches.
"""

import logging
import os

import numpy as np

from yass.batch import BatchProcessor


# configure logging to get information about the process
logging.basicConfig(level=logging.INFO)


# raw data file
path_to_neuropixel_data = (os.path.expanduser('~/data/ucl-neuropixel'
                           '/rawDataSample.bin'))


# on each batch, we find the maximum value in every channel
def max_in_channel(batch):
    """Add one to every element in the batch
    """
    return np.max(batch, axis=0)


# create batch processor for the data
bp = BatchProcessor(path_to_neuropixel_data,
                    dtype='int16', n_channels=385, data_format='wide',
                    max_memory='10MB')

# appply a multi channel transformation, each batch will be a temporal
# subset with observations from all selected n_channels, the size
# of the subset is calculated depending on max_memory. Results
# from every batch are returned in a list
res = bp.multi_channel_apply(max_in_channel,
                             mode='memory',
                             channels='all')


# we have 8 batches, so our list contains 8 elements
len(res)


# output for the first batch
res[0]


# stack results from every batch
arr = np.stack(res, axis=0)


# let's find the maximum value along every channel in all the dataset
np.max(arr, axis=0)
