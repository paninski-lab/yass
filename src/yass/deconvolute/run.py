import numpy as np
import logging

from .deconvolute import Deconvolution
from .. import read_config


def run(spike_train_clear, templates, spike_index_collision):
    """Run deconvolution

    Parameters
    ----------
    spike_train
    spikes_left
    templates

    Returns
    -------
    spike_train

    Examples
    --------

    .. literalinclude:: ../examples/deconvolute.py
    """
    logger = logging.getLogger(__name__)

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index_collision.shape: {}'
                  .format(templates.shape, spike_index_collision.shape))

    CONFIG = read_config()
    deconv = Deconvolution(CONFIG, np.transpose(templates, [1, 0, 2]),
                           spike_index_collision, filename='whiten.bin')
    spike_train_deconv = deconv.fullMPMU()

    logger.debug('spike_train_deconv.shape: {}'
                 .format(spike_train_deconv.shape))

    spike_train = np.concatenate((spike_train_deconv, spike_train_clear))

    idx_sort = np.argsort(spike_train[:, 0])
    spike_train = spike_train[idx_sort]

    logger.debug('spike_train.shape: {}'.format(spike_train.shape))

    idx_keep = np.zeros(spike_train.shape[0], 'bool')

    for k in range(templates.shape[2]):
        idx_c = np.where(spike_train[:, 1] == k)[0]
        idx_keep[idx_c[np.concatenate(([True], np.diff(spike_train[idx_c,0]) > 1))]] = 1

    spike_train = spike_train[idx_keep]

    logger.debug('deduplicated spike_train.shape: {}'
                 .format(spike_train.shape))

    return spike_train
