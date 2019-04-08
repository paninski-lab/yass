import os
import logging
import numpy as np

from yass import read_config
from yass.reader import READER
from yass.merge.util import (partition_input,
                             merge_units,
                             align_templates)
from yass.merge.merge import TemplateMerge

def run(output_directory,
        raw_data,
        fname_spike_train,
        fname_templates,
        fname_up=None,
        fname_recording=None,
        recording_dtype=None,
        fname_residual=None,
        residual_dtype=None):

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    fname_spike_train_out = os.path.join(output_directory,
                                         'spike_train.npy')
    fname_templates_out = os.path.join(output_directory,
                                       'templates.npy')

    if os.path.exists(fname_spike_train_out) and os.path.exists(fname_templates_out):
        return fname_templates_out, fname_spike_train_out

    # partition spike_idnex_chunk using the second column
    partition_dir = os.path.join(output_directory, 'input_partition')
    fnames_input = partition_input(partition_dir,
                                   fname_templates,
                                   fname_spike_train,
                                   fname_up)

    # get reader
    if raw_data:
        reader = READER(fname_recording,
                        recording_dtype,
                        CONFIG)
    else:
        reader = READER(fname_residual,
                        residual_dtype,
                        CONFIG)

    # initialize merge: find candidates
    tm = TemplateMerge(
        output_directory,
        raw_data,
        reader,
        fname_templates,
        fnames_input,
        CONFIG.resources.multi_processing,
        CONFIG.resources.n_processors)

    # find merge pairs
    merge_pairs = tm.get_merge_pairs()
    # update templates adn spike train accordingly
    spike_train_new, templates_new, merge_array = merge_units(
        fname_templates, fname_spike_train, merge_pairs)
    
    # save result
    fname_merge_pairs = os.path.join(output_directory,
                                     'merge_pairs.npy')
    fname_merge_array = os.path.join(output_directory,
                                     'merge_array.npy')
    np.save(fname_merge_pairs, merge_pairs)
    np.save(fname_merge_array, merge_array)    
    np.save(fname_spike_train_out, spike_train_new)
    np.save(fname_templates_out, templates_new)

    logger.info('Number of units after merge: {}'.format(templates_new.shape[0]))

    return fname_templates_out, fname_spike_train_out
