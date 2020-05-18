"""
Neuropixel spike sorting: http://data.cortexlab.net/singlePhase3/
"""
import os.path as path
import numpy as np

# download rawDataSample.bin, channel_map.npy and channel_positions.npy
# from here: http://data.cortexlab.net/singlePhase3/data/
ROOT = '/Users/Edu/data/ucl-neuropixel/'

channel_map = np.load(path.join(ROOT, 'channel_map.npy')).reshape(374)
channel_positions = np.load(path.join(ROOT, 'channel_positions.npy'))
raw = np.fromfile(path.join(ROOT, 'rawDataSample.bin'),
                  dtype='int16').reshape((385, -1))
raw = raw[channel_map, :]

raw.tofile(path.join(ROOT, 'neuropixel.bin'))
