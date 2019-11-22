import os
import logging
import numpy as np
from scipy.spatial.distance import pdist, squareform

from yass import read_config
from yass.reader import READER
from yass.merge.util import (partition_input,
                             merge_units)
from yass.merge.merge import TemplateMerge
from yass.augment.noise import kill_signal, search_noise_snippets

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
        spatial_cov, temporal_cov = get_covariance(reader_residual, CONFIG)
        np.save(fname_spatial_cov, spatial_cov)
        np.save(fname_temporal_cov, temporal_cov)

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


def get_covariance(reader, CONFIG):
    
    # get data chunk
    chunk_5sec = 5*CONFIG.recordings.sampling_rate
    if CONFIG.rec_len < chunk_5sec:
        chunk_5sec = CONFIG.rec_len
    small_batch = reader.read_data(
    data_start=CONFIG.rec_len//2 - chunk_5sec//2,
    data_end=CONFIG.rec_len//2 + chunk_5sec//2)
    
    # get noise floor of recording
    noised_killed, is_noise_idx = kill_signal(small_batch, 3, CONFIG.spike_size)
    
    # spatial covariance
    spatial_cov_all = np.divide(np.matmul(noised_killed.T, noised_killed),
                        np.matmul(is_noise_idx.T, is_noise_idx))
    sig = np.sqrt(np.diag(spatial_cov_all))
    spatial_cov_all = spatial_cov_all/(sig[:,None]*sig[None])

    chan_dist = squareform(pdist(CONFIG.geom))
    chan_dist_unique = np.unique(chan_dist)
    cov_by_dist = np.zeros(len(chan_dist_unique))
    for ii, d in enumerate(chan_dist_unique):
        cov_by_dist[ii] = np.mean(spatial_cov_all[chan_dist == d])
    dist_in = cov_by_dist > 0.1
    chan_dist_unique = chan_dist_unique[dist_in]
    cov_by_dist = cov_by_dist[dist_in]
    spatial_cov = np.vstack((cov_by_dist, chan_dist_unique)).T

    # get noise snippets
    noise_wf = search_noise_snippets(
    noised_killed, is_noise_idx, 1000,
    CONFIG.spike_size,
    channel_choices=None,
    max_trials_per_sample=100,
    allow_smaller_sample_size=True)

    # get temporal covariance
    temp_cov = np.cov(noise_wf.T)
    sig = np.sqrt(np.diag(temp_cov))
    temp_cov = temp_cov/(sig[:,None]*sig[None])
    
    #w, v = np.linalg.eig(temp_cov)
    #inv_half_cov = np.matmul(np.matmul(v, np.diag(1/np.sqrt(w))), v.T)
    
    return spatial_cov, temp_cov
