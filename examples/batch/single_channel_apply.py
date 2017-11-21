import os

from yass.batch.new import BatchProcessor


path_to_neuropixel_data = (os.path.expanduser('~/data/ucl-neuropixel'
                           '/rawDataSample.bin'))
path_to_filtered_data = (os.path.expanduser('~/data/ucl-neuropixel'
                         '/tmp/filtered.bin'))


bp = BatchProcessor(path_to_neuropixel_data,
                    dtype='int16', n_channels=385, data_format='wide',
                    max_memory='1MB')



bp_long = BatchProcessor(path_to_long,
                         dtype='int64', n_channels=50, data_format='long',
                         max_memory='500MB')

path = bp_long.single_channel_apply(dummy, path_to_out)

