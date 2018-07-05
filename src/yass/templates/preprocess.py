import os
import logging

import numpy as np

from yass.templates.util import get_templates
from yass.templates.choose import choose_templates
from yass.templates.crop import crop_and_align_templates


def preprocess(CONFIG, spike_train, path_to_data, chosen_templates_indexes):
    """Read, choose, crop and align templates from a spike_train
    """
    logger = logging.getLogger(__name__)

    # make sure standarized data already exists
    if not os.path.exists(path_to_data):
        raise ValueError('Standarized data does not exist in: {}, this is '
                         'needed to generate training data, run the '
                         'preprocesor first to generate it'
                         .format(path_to_data))

    n_spikes, _ = spike_train.shape

    logger.info('Getting templates...')

    # add weight of one to every spike
    weighted_spike_train = np.hstack((spike_train,
                                      np.ones((n_spikes, 1), 'int32')))

    # get templates
    templates_uncropped, _ = get_templates(weighted_spike_train,
                                           path_to_data,
                                           CONFIG.resources.max_memory,
                                           4*CONFIG.spike_size)

    templates_uncropped = np.transpose(templates_uncropped, (2, 1, 0))

    logger.debug('Uncropped templates  shape: {}'
                 .format(templates_uncropped.shape))

    # choose good templates (user selected and amplitude above threshold)
    # TODO: maybe the minimum_amplitude parameter should be selected by the
    # user
    templates_uncropped = choose_templates(templates_uncropped,
                                           chosen_templates_indexes,
                                           minimum_amplitude=4)

    if templates_uncropped.shape[0] == 0:
        raise ValueError("Coulndt find any good templates...")

    logger.debug('Uncropped templates shape after selection: {}'
                 .format(templates_uncropped.shape))

    templates = crop_and_align_templates(templates_uncropped,
                                         CONFIG.spike_size,
                                         CONFIG.neigh_channels,
                                         CONFIG.geom)

    logger.debug('Templates shape after crop and align %s', templates.shape)

    return templates, templates_uncropped
