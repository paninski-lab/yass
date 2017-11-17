import os

import numpy as np
from yass.batch import RecordingsReader

# generate some big files
output_folder = '/Users/Edu/data/yass-benchmarks'
wide_data = np.random.rand(50, 1000000)
long_data = np.random.rand(1000000, 50)

path_to_wide = os.path.join(output_folder, 'wide.bin')
path_to_long = os.path.join(output_folder, 'long.bin')

wide_data.tofile(path_to_wide)
long_data.tofile(path_to_long)

# load the files using the readers, they are agnostic on the data type
# and will behave exactly the same
reader_wide = RecordingsReader(path_to_wide, dtype='float64',
                               channels=50, data_format='wide')

reader_long = RecordingsReader(path_to_long, dtype='float64',
                               channels=50, data_format='long')


# first index is for observations and second index for channels
reader_wide[10000:20000, 20:30]
reader_wide.shape

# same applies even if your data is in 'long' format, first index for
# observations, second for channels, the output is converted to 'wide'
reader_long[10000:20000:, 20:30]
reader_long.shape
