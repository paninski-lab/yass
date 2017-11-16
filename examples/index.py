import logging

from yass.batch import BatchIndexer

logging.basicConfig(level=logging.INFO)

bi = BatchIndexer(10000000, 512, 'float64', '1GB')

indexer = bi.channelwise()

[idx for idx in indexer]

indexer = bi.temporalwise()

[idx for idx in indexer]