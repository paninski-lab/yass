"""
%load_ext autoreload
%autoreload 2
"""

import logging

from yass.batch import IndexGenerator

logging.basicConfig(level=logging.INFO)

ig = IndexGenerator(1000, 10, 'int16', 124)
indexer = ig.temporalwise(to_time=1000)
[idx for idx in indexer]


indexer = ig.channelwise(from_time=1)
[idx for idx in indexer]


indexer = ig.channelwise(from_time=1, complete_channel_batch=False)
x = [idx for idx in indexer]
x
[list(y) for y in x]


indexer = ig.channelwise(from_time=10, to_time=20)
[idx for idx in indexer]

indexer = ig.channelwise(to_time=20)
[idx for idx in indexer]
