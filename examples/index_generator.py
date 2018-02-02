"""
%load_ext autoreload
%autoreload 2
"""
import logging

from yass.batch import IndexGenerator

logging.basicConfig(level=logging.ERROR)

# init indexer for a matrix with int16, 1 million observations and
# 512 columns with maximum memory usage of 124 bytes
ig = IndexGenerator(1000000, 512, 'int16', '10MB')


# index every channel and every observation, each index object will correspond
# to a single channel
indexer = ig.channelwise()

# index observations from 100 to 200, each index will correspond to a single
# channel
indexer = ig.channelwise(complete_channel_batch=False,
                         from_time=100, to_time=200)

# samme but only go trough channel 10
indexer = ig.channelwise(complete_channel_batch=False,
                         from_time=100, to_time=200,
                         channels=10)

# samme but only go trough channels 10, 20
indexer = ig.channelwise(complete_channel_batch=False,
                         from_time=100, to_time=200,
                         channels=[10, 20])

# new indexed, this time with 1MB max memory
ig = IndexGenerator(1000000, 512, 'int16', '1MB')

# since all 1 million observations per channel do not fit in 1MB we need to
# turn of complete_channel_batch
indexer = ig.channelwise(complete_channel_batch=False)
