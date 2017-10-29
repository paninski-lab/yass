"""
Using the indexer module to read multi-channel recordings
"""
from yass import Indexer


# initialize indexer
indexer = Indexer('path/to/data.bin', n_channels=50,
                  mode='long', dtype='float64')

# read observations from 1000 to 2000 in channels 1 and 5
indexer.read(observations=(1000, 2000), channels=(1, 5))

# read observations from 1000 to 2000 in all channels
indexer.read(observations=(1000, 2000), channels='all')
