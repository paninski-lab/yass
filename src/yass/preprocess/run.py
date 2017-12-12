"""
Preprocess pipeline
"""

import datetime
import logging
import os.path

import numpy as np
import yaml

from .. import read_config
from ..batch import BatchPipeline, BatchProcessor, RecordingsReader
from ..batch import PipedTransformation as Transform

from .filter import butterworth
from .standarize import standarize
from .whitening import whitening_matrix, whitening, localized_whitening_matrix, whitening_score
from .detect import threshold_detection
from .score import get_score_pca, get_pca_suff_stat, get_pca_projection
from ..neuralnetwork import NeuralNetDetector, NeuralNetTriage, nn_detection


def run():
    """Execute preprocessing pipeline

    Returns
    -------
    score: list
        List of size n_channels, each list contains a (clear spikes x
        number of features x number of channels) multidimensional array
        score for every clear spike

    clear_index: list
        List of size n_channels, each list contains the indexes in
        spike_times (first column) where the spike was clear

    spike_times: list
        List with n_channels elements, each element contains spike times
        in the first column and [SECOND COLUMN?]

    Examples
    --------

    .. literalinclude:: ../examples/preprocess.py
    """

    # logger = logging.getLogger(__name__)

    CONFIG = read_config()

    tmp = os.path.join(CONFIG.data.root_folder, 'tmp')

    if not os.path.exists(tmp):
        os.makedirs(tmp)

    path = os.path.join(CONFIG.data.root_folder, CONFIG.data.recordings)
    dtype = CONFIG.recordings.dtype

    # initialize pipeline object, one batch per channel
    pipeline = BatchPipeline(path, dtype, CONFIG.recordings.n_channels,
                             CONFIG.recordings.format,
                             CONFIG.resources.max_memory, tmp,
                             mode='single_channel_one_batch')

    # add filter transformation if necessary
    if CONFIG.preprocess.filter:
        filter_op = Transform(butterworth,
                              'filtered.bin', keep=True,
                              low_freq=CONFIG.filter.low_pass_freq,
                              high_factor=CONFIG.filter.high_factor,
                              order=CONFIG.filter.order,
                              sampling_freq=CONFIG.recordings.sampling_rate)

        pipeline.add([filter_op])

    # standarize
    standarize_op = Transform(standarize, 'standarized.bin',
                              keep=True,
                              srate=CONFIG.recordings.sampling_rate)

    pipeline.add([standarize_op])

    # whiten

    pipeline.run()
