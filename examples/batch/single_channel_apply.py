import logging
import os

from yass.batch.new import BatchProcessor
from yass.batch import RecordingsReader
from yass.preprocess.filter import butterworth

logging.basicConfig(level=logging.INFO)


path_to_neuropixel_data = (os.path.expanduser('~/data/ucl-neuropixel'
                           '/rawDataSample.bin'))
path_to_filtered_data = (os.path.expanduser('~/data/ucl-neuropixel'
                         '/tmp/filtered.bin'))


bp = BatchProcessor(path_to_neuropixel_data,
                    dtype='int16', n_channels=385, data_format='wide',
                    max_memory='500MB')


bp.single_channel_apply(butterworth, path_to_filtered_data,
                        low_freq=300, high_factor=0.1,
                        order=3, sampling_freq=30000)

filtered = RecordingsReader(path_to_filtered_data)
