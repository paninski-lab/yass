"""
Generates the sample data
"""
from pathlib import Path

import numpy as np

from yass import preprocess
from yass.batch import RecordingsReader

SIZE = 500000


DATA_FOLDER = Path('~', 'data').expanduser()
OUTPUT_FOLDER = Path(DATA_FOLDER, 'sample-data')


# retinal, 49ch
path_to_retinal_folder = Path(DATA_FOLDER, 'yass')
path_to_retinal_data = Path(path_to_retinal_folder, 'ej49_data1_set1.bin')
path_to_retinal_geom = Path(path_to_retinal_folder, 'ej49_geometry1.txt')


retinal = RecordingsReader(str(path_to_retinal_data),
                           dtype='int16', n_channels=49,
                           data_order='samples', loader='array').data

retinal_sub = retinal[:SIZE, :]

retinal_sub.tofile(str(Path(OUTPUT_FOLDER, 'retinal.bin')))


# load data and sample


# CHANNELS = 30
# ch = np.load(path.join(PATH_TO_DATA, 'channel_positions.npy'))
# n_ch, _ = ch.shape

# d = np.fromfile(path.join(PATH_TO_DATA, 'rawDataSample.bin'), dtype='int16')
# d = d.reshape((385, 1800000))

# sample = d[:CHANNELS, :SIZE].T
# sample.tofile('examples/data/neuropixel.bin')

# np.save('examples/data/neuropixel_channels.npy', ch[:CHANNELS, :])


# preprocess.butterworth('examples/data/neuropixel.bin',
#                        low_frequency=300,
#                        high_factor=0.1,
#                        order=3,
#                        sampling_frequency=30000,
#                        dtype='int16',
#                        n_channels=CHANNELS,
#                        data_order='samples',
#                        max_memory='1GB',
#                        output_path='examples/data',
#                        output_dtype='float16',
#                        if_file_exists='overwrite')


# preprocess.standarize('examples/data/filtered.bin',
#                       dtype='float16',
#                       n_channels=CHANNELS,
#                       data_order='samples',
#                       sampling_frequency=30000,
#                       max_memory='1GB',
#                       output_path='examples/data',
#                       output_dtype='float16',
#                       if_file_exists='overwrite')
