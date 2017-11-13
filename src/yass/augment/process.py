import datetime
import logging
import os.path
import numpy as np

from .. import read_config
from ..batch import BatchProcessorFactory

from ..preprocess.filter import whitening_matrix, whitening, butterworth
from ..preprocess.standarize import standarize, sd

def process_data(CONFIG):
    
    logger = logging.getLogger(__name__)
    
    # initialize processor for raw data
    path = os.path.join(CONFIG.root, CONFIG.filename)
    dtype = CONFIG.dtype

    # initialize factory
    factory = BatchProcessorFactory(path_to_file=None,
                                    dtype=None,
                                    n_channels=CONFIG.nChan,
                                    max_memory=CONFIG.maxMem,
                                    buffer_size=None)

    if CONFIG.doFilter == 1:
        # make batch processor for raw data -> buterworth -> filtered
        bp = factory.make(path_to_file=path, dtype=dtype,
                          buffer_size=0)
        logger.info('Initialized butterworth batch processor: {}'
                    .format(bp))

        # run filtering
        path = os.path.join(CONFIG.root,  'tmp/filtered.bin')
        dtype = bp.process_function(butterworth,
                                    path,
                                    CONFIG.filterLow,
                                    CONFIG.filterHighFactor,
                                    CONFIG.filterOrder,
                                    CONFIG.srate)

    # TODO: cache computations
    # make batch processor for filtered -> standarize -> standarized
    bp = factory.make(path_to_file=path, dtype=dtype, buffer_size=0)

    # compute the standard deviation using the first batch only
    batch1 = next(bp)
    sd_ = sd(batch1, CONFIG.srate)

    # make another batch processor
    bp = factory.make(path_to_file=path, dtype=dtype, buffer_size=0)
    logger.info('Initialized standarization batch processor: {}'
                .format(bp))

    # run standarization
    path = os.path.join(CONFIG.root,  'tmp/standarized.bin')
    dtype = bp.process_function(standarize,
                                path,
                                sd_)