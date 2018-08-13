"""
Generates the sample data
"""
import os.path as path

import numpy as np

from yass import preprocess

SIZE = 200000
CHANNELS = 30

PATH_TO_DATA = path.join(path.expanduser('~'), 'data/sample-data')

ch = np.load(path.join(PATH_TO_DATA, 'channel_positions.npy'))
n_ch, _ = ch.shape

d = np.fromfile(path.join(PATH_TO_DATA, 'rawDataSample.bin'), dtype='int16')
d = d.reshape((385, 1800000))

sample = d[:CHANNELS, :SIZE].T
sample.tofile('examples/data/neuropixel.bin')

np.save('examples/data/neuropixel_channels.npy', ch[:CHANNELS, :])


preprocess.butterworth('examples/data/neuropixel.bin',
                       low_frequency=300,
                       high_factor=0.1,
                       order=3,
                       sampling_frequency=30000,
                       dtype='int16',
                       n_channels=CHANNELS,
                       data_order='samples',
                       max_memory='1GB',
                       output_path='examples/data',
                       output_dtype='float16',
                       if_file_exists='overwrite')


preprocess.standarize('examples/data/filtered.bin',
                      dtype='float16',
                      n_channels=CHANNELS,
                      data_order='samples',
                      sampling_frequency=30000,
                      max_memory='1GB',
                      output_path='examples/data',
                      output_dtype='float16',
                      if_file_exists='overwrite')
