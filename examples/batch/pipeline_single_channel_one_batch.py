import logging
import os

import matplotlib.pyplot as plt

from yass.batch.pipeline import BatchPipeline, PipedTransformation
from yass.batch import RecordingsReader
from yass.preprocess.filter import butterworth_single_channel
from yass.preprocess.standarize import _standarize

logging.basicConfig(level=logging.DEBUG)


path_to_neuropixel_data = (os.path.expanduser('~/data/ucl-neuropixel'
                           '/rawDataSample.bin'))
path_output = os.path.expanduser('~/data/ucl-neuropixel/tmp')


pipeline = BatchPipeline(path_to_neuropixel_data, dtype='int16',
                         n_channels=385, data_format='wide',
                         max_memory='500MB',
                         mode='single_channel_one_batch',
                         from_time=None, to_time=None, channels='all',
                         output_path=path_output)

butterworth = PipedTransformation(butterworth_single_channel, 'filtered.bin',
                                  keep=True, low_freq=300, high_factor=0.1,
                                  order=3, sampling_freq=30000)

standarize = PipedTransformation(_standarize, 'standarized.bin',
                                 keep=True, srate=30000)

pipeline.add([butterworth, standarize])

pipeline.run()


raw = RecordingsReader(path_to_neuropixel_data, dtype='int16',
                       n_channels=385, data_format='wide')
filtered = RecordingsReader(os.path.join(path_output, 'filtered.bin'))
standarized = RecordingsReader(os.path.join(path_output, 'standarized.bin'))

# plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(raw[:2000, 0])
ax1.set_title('Raw data')
ax2.plot(filtered[:2000, 0])
ax2.set_title('Filtered data')
ax3.plot(standarized[:2000, 0])
ax3.set_title('Standarized data')
plt.tight_layout()
plt.show()
