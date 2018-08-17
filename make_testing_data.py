"""
Generates data for testing
"""
from pathlib import Path

import numpy as np

from yass.preprocess.filter import butterworth
from yass.batch import RecordingsReader


path_to_data_storage = Path('~', 'data').expanduser()
path_to_examples_data = Path('examples', 'data')
path_to_tests_data = Path('tests', 'data')


SIZE = 500000
OUTPUT_FOLDER = Path(path_to_data_storage, 'sample-data')
# retinal, 49ch
path_to_retinal_folder = Path(path_to_data_storage, 'yass')
path_to_retinal_data = Path(path_to_retinal_folder, 'ej49_data1_set1.bin')
path_to_retinal_geom = Path(path_to_retinal_folder, 'ej49_geometry1.txt')


retinal = RecordingsReader(str(path_to_retinal_data),
                           dtype='int16', n_channels=49,
                           data_order='samples', loader='array').data

retinal_sub = retinal[:SIZE, :]

retinal_sub.tofile(str(Path(OUTPUT_FOLDER, 'retinal.bin')))

# Dataset 2: neuropixel
seconds = 5
channels = 10
output_name_data = 'neuropixel.bin'
output_name_geometry = 'neuropixel_channels.npy'

path_to_neuro_data = Path(path_to_data_storage, 'neuro')
dtype = 'int16'
data_order = 'samples'
sampling_frequency = 30000
observations = sampling_frequency * seconds

data = np.fromfile(str(Path(path_to_neuro_data, 'rawDataSample.bin')),
                   dtype='int16')

geometry = np.load(str(Path(path_to_neuro_data, 'channel_positions.npy')))
n_ch, _ = geometry.shape


data = data.reshape((385, 1800000)).T

sample_data = data[:observations, :channels].T
sample_geometry = geometry[:channels, :]

# save data to examples/data and tests/data folders
sample_data.tofile(str(Path(path_to_examples_data, output_name_data)))
sample_data.tofile(str(Path(path_to_tests_data, output_name_data)))

# save geometry data
np.save(str(Path(path_to_examples_data, output_name_geometry)),
        sample_geometry)
np.save(str(Path(path_to_tests_data, output_name_geometry)),
        sample_geometry)


butterworth(str(Path(path_to_tests_data, output_name_data)),
            dtype=dtype,
            n_channels=channels, data_order=data_order,
            order=3, low_frequency=300, high_factor=0.1,
            sampling_frequency=sampling_frequency, max_memory='1GB',
            output_path=str(path_to_tests_data),
            standarize=True,
            output_filename='standarized.bin',
            if_file_exists='overwrite',
            output_dtype='float32')


# import matplotlib.pyplot as plt
# standarized = RecordingsReader('tests/data/standarized.bin', loader='array').data
# plt.plot(standarized[1000:1200])
# plt.show()
