import os
import logging
import numpy as np

from yass import read_config
from yass.reader import READER
from yass.merge.util import (partition_input,
                             merge_units)
from yass.merge.merge import TemplateMerge
from yass.noise import get_noise_covariance

def run(output_directory,
        fname_spike_train,
        fname_shifts,
        fname_scales,
        fname_templates,
        fname_soft_assignment,
        fname_residual,
        residual_dtype):

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    fname_spike_train_out = os.path.join(output_directory,
                                         'spike_train.npy')
    fname_templates_out = os.path.join(output_directory,
                                       'templates.npy')
    fname_soft_assignment_out = os.path.join(output_directory,
                                       'soft_assignment.npy')

    if os.path.exists(fname_spike_train_out) and os.path.exists(fname_templates_out):
        return (fname_templates_out,
                fname_spike_train_out,
                fname_soft_assignment_out)

    reader_residual = READER(fname_residual,
                             residual_dtype,
                             CONFIG)

    # get whitening filters
    fname_spatial_cov = os.path.join(output_directory, 'spatial_cov.npy')
    fname_temporal_cov = os.path.join(output_directory, 'temporal_cov.npy')
    if not (os.path.exists(fname_spatial_cov) and os.path.exists(fname_temporal_cov)):
        spatial_cov, temporal_cov = get_noise_covariance(reader_residual, CONFIG)
        np.save(fname_spatial_cov, spatial_cov)
        np.save(fname_temporal_cov, temporal_cov)
    else:
        spatial_cov = np.load(fname_spatial_cov)
        temporal_cov = np.load(fname_temporal_cov)

    # initialize merge: find candidates
    logger.info("finding merge candidates")
    tm = TemplateMerge(
        output_directory,
        reader_residual,
        fname_templates,
        fname_spike_train,
        fname_shifts,
        fname_scales,
        fname_soft_assignment,
        fname_spatial_cov,
        fname_temporal_cov,
        CONFIG.geom,
        CONFIG.resources.multi_processing,
        CONFIG.resources.n_processors)

    # find merge pairs
    logger.info("merging pairs")
    tm.get_merge_pairs()

    # update templates adn spike train accordingly
    logger.info("udpating templates and spike train")
    (templates_new,
     spike_train_new,
     soft_assignment_new,
     merge_array) = tm.merge_units()

    # save results
    fname_merge_array = os.path.join(output_directory,
                                     'merge_array.npy')
    np.save(fname_merge_array, merge_array)    
    np.save(fname_spike_train_out, spike_train_new)
    np.save(fname_templates_out, templates_new)
    np.save(fname_soft_assignment_out, soft_assignment_new)

    logger.info('Number of units after merge: {}'.format(templates_new.shape[0]))

    return (fname_templates_out,
            fname_spike_train_out,
            fname_soft_assignment_out)
