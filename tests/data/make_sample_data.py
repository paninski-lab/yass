"""
Run this from the yass folder
"""
import os.path as path
import numpy as np
from yass import preprocess
import yass

PATH_TO_DATA = path.join(path.expanduser('~'), 'data/ucl-neuropixel')

ch = np.load(path.join(PATH_TO_DATA, 'channel_positions.npy'))
n_ch, _ = ch.shape

d = np.fromfile(path.join(PATH_TO_DATA, 'rawDataSample.bin'), dtype='int16')
d = d.reshape((385, 1800000))

# save sample data
sample = d[:10, :10000].T
sample.tofile('tests/data/neuropixel.bin')

# save geometry
np.save('tests/data/neuropixel_channels.npy', ch[:10, :])

yass.set_config('tests/config_nnet.yaml')

_ = preprocess.run('sample_pipeline_output')
