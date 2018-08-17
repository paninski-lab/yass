"""
Sample data for testing
"""
import shutil
from pathlib import Path

import numpy as np

from yass.preprocess.filter import butterworth
from yass.batch import RecordingsReader
from yass import geometry as yass_geometry


path_to_data_storage = Path('~', 'data').expanduser()

path_to_neuro_folder = Path(path_to_data_storage, 'neuro')
path_to_neuro_data = Path(path_to_neuro_folder, 'rawDataSample.bin')
path_to_neuro_geom = Path(path_to_neuro_folder, 'channel_positions.npy')


path_to_retina_folder = Path(path_to_data_storage, 'retina')
path_to_retina_data = Path(path_to_retina_folder, 'ej49_data1_set1.bin')
path_to_retina_geom = Path(path_to_retina_folder, 'ej49_geometry1.txt')


path_to_output_folder = Path(path_to_data_storage, 'yass-testing-data')
path_to_output_folder_neuro = Path(path_to_output_folder, 'neuropixel')
path_to_output_folder_retina = Path(path_to_output_folder, 'retina')

if path_to_output_folder.exists():
    print('Output Folder exists, removing contents....')
    shutil.rmtree(str(path_to_output_folder))


path_to_output_folder.mkdir()
path_to_output_folder_neuro.mkdir()
path_to_output_folder_retina.mkdir()


# Dataset 1: retina
seconds = 60
channels = 49

dtype = 'int16'
data_order = 'samples'
sampling_frequency = 20000
observations = sampling_frequency * seconds


# retina, 49ch
retina = RecordingsReader(str(path_to_retina_data),
                          dtype='int16', n_channels=49,
                          data_order='samples', loader='array').data

sample_data = retina[:observations, :]
sample_data.tofile(str(Path(path_to_output_folder_retina, 'data.bin')))

geometry = yass_geometry.parse(str(path_to_retina_geom), 49)
sample_geometry = geometry[:channels, :]
np.save(str(Path(path_to_output_folder_retina, 'geometry.npy')),
        sample_geometry)


# Dataset 2: neuropixel
seconds = 5
channels = 10
dtype = 'int16'
data_order = 'samples'
sampling_frequency = 30000
observations = sampling_frequency * seconds

data = np.fromfile(str(Path(path_to_neuro_data)), dtype=dtype)
data = data.reshape((385, 1800000)).T
sample_data = data[:observations, :channels]

geometry = np.load(str(Path(path_to_neuro_geom)))
sample_geometry = geometry[:channels, :]

# save data and geometry
sample_data.tofile(str(Path(path_to_output_folder_neuro, 'data.bin')))
np.save(str(Path(path_to_output_folder_neuro, 'geometry.npy')),
        sample_geometry)


butterworth(str(Path(path_to_output_folder_neuro, 'data.bin')),
            dtype=dtype,
            n_channels=channels, data_order=data_order,
            order=3, low_frequency=300, high_factor=0.1,
            sampling_frequency=sampling_frequency, max_memory='1GB',
            output_path=str(path_to_output_folder_neuro),
            standarize=True,
            output_filename='standarized.bin',
            if_file_exists='overwrite',
            output_dtype='float32')


# import matplotlib.pyplot as plt
# standarized = RecordingsReader('tests/data/standarized.bin', loader='array').data
# plt.plot(standarized[1000:1200])
# plt.show()
