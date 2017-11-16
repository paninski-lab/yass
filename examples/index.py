import logging

from yass.batch import BatchIndexer

logging.basicConfig(level=logging.INFO)

bi = BatchIndexer(10000000, 512, 'float64', '1GB')


indexer = bi.channelwise()
[idx for idx in indexer]

indexer = bi.channelwise(from_time=10, to_time=20)
[idx for idx in indexer]

indexer = bi.channelwise(to_time=20)
[idx for idx in indexer]

indexer = bi.temporalwise(to_time=10000000)

[idx for idx in indexer]