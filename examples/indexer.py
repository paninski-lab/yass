"""
Using the indexer module to read multi-channel recordings
"""
from yass import Indexer


# initialize indexer
indexer = Indexer('path/to/data.bin', n_channels=50,
                  mode='long', dtype='float64')

# read some data
indexer.read()
