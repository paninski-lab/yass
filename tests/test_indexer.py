from yass.index import Indexer

indexer = Indexer('data.bin', n_channels=10, mode='long', dtype='int16')

indexer.read(observations=(100, 200), channels='all')

import itertools
from operator import itemgetter

pairs = itertools.product(range(10), range(10000))
wide_pairs = sorted(pairs, key=itemgetter(0))
long_pairs = sorted(wide_pairs, key=itemgetter(1))

wide_values = [float('{}.{}'.format(channel, obs)) for channel, obs in wide_pairs]
long_values = [float('{}.{}'.format(channel, obs)) for channel, obs in long_pairs]

long_data = np.array(long_values).reshape((10000, 10))
wide_data = np.array(wide_values).reshape((10, 10000))

long_data.tofile('long.bin')
wide_data.tofile('wide.bin')

indexer = Indexer('long.bin', n_channels=10, mode='long', dtype='float64')
indexer.read(observations=(1000, 1020), channels=(1, 5))

indexer = Indexer('wide.bin', n_channels=10, mode='wide', dtype='float64')
indexer.read(observations=(1000, 1020), channels=(1, 5))
