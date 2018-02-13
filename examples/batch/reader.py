# coding: utf-8

# See notebook:
# https://github.com/paninski-lab/yass-examples/blob/master/batch/reader.ipynb

"""
Reading large files with RecordingsReader
"""

import os

import numpy as np
from yass.batch import RecordingsReader


# generate data
output_folder = os.path.join(os.path.expanduser('~'), 'data/yass')
wide_data = np.random.rand(50, 100000)
long_data = wide_data.T

path_to_wide = os.path.join(output_folder, 'wide.bin')
path_to_long = os.path.join(output_folder, 'long.bin')

wide_data.tofile(path_to_wide)
long_data.tofile(path_to_long)


# load the files using the readers, they are agnostic on the data shape
# and will behave exactly the same
reader_wide = RecordingsReader(path_to_wide, dtype='float64',
                               n_channels=50, data_format='wide')

reader_long = RecordingsReader(path_to_long, dtype='float64',
                               n_channels=50, data_format='long')


reader_wide.shape, reader_long.shape


# first index is for observations and second index for channels
obs = reader_wide[10000:20000, 20:30]
obs, obs.shape


# same applies even if your data is in 'wide' shape, first index for
# observations, second for channels, the output is converted to 'long'
# by default but you can change it
obs = reader_long[10000:20000, 20:30]
obs, obs.shape
