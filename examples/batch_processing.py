import numpy as np

from yass.batch import BatchProcessor

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

data = np.empty((0, n_channels))

# to batch processor is an interator so you can use it in a for loop
# file is opened and closed automatically
for i, batch in enumerate(batch_processor):
    print('Processing batch {}...'.format(i + 1))

    # sum one to every entry in the batch and append it to data
    data = np.vstack((data, batch + 1))

print(batch_processor)

# you can also use BatchProcessor batch by batch
batch_processor = BatchProcessor(path, dtype=dtype, n_channels=n_channels,
                                 max_memory=140000,
                                 buffer_size=0)

batch_1 = next(batch_processor)
batch_2 = next(batch_processor)

# or load a batch directly...
batch_10 = batch_processor.load_batch(10)
