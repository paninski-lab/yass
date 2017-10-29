"""
Using the indexer module to read multi-channel recordings
"""
from yass import indexer

import numpy as np

shape = (10000000, 10)
big_matrix = np.array(range(100000000)).reshape(shape)
big_matrix.tofile('data.bin')


data = np.memmap('data.bin', dtype='int64', shape=shape)

indexer.read(observations=(1000, 2000), channels=(1, 5, 6))