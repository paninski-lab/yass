import numpy as np

from yass.batch import BatchProcessor
from yass.preprocess.filter import butterworth

dtype = 'int16'
dsize = np.dtype(dtype).itemsize
obs_total = 100000
n_channels = 7
obs_size = dsize * n_channels
path = '../tests/sample_data/sample_100k.bin'


batch_processor = BatchProcessor(path, dtype=dtype, n_channels=n_channels,
                                 max_memory=140000,
                                 buffer_size=0)

# you can get some information printing the object
print(batch_processor)


dtype = batch_processor.process_function(butterworth,
                                         path_to_file='../tests/filtered.bin',
                                         low_freq=300,
                                         high_factor=0.1,
                                         order=3,
                                         sampling_freq=20000)

print(batch_processor)
