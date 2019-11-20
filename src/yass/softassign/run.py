#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:44:54 2019

@author: kevin
"""

import os
import logging
import numpy as np
from scipy.stats import chi2
from yass import read_config
from yass.reader import READER
from yass.softassign.template_soft_assignment import TEMPLATE_ASSIGN_OBJECT
from yass.merge.run import get_covariance

def run(template_fname,
        spike_train_fname,
        shifts_fname,
        output_directory,
        residual_fname,
        residual_dtype, 
        window_size, 
        similarity_threshold, 
        similar_units):

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    #
    prob_fname = os.path.join(output_directory, 'template_soft_assignment.npy')
    outlier_fname = os.path.join(output_directory, 'outliers.npy')
    logprobs_fname = os.path.join(output_directory, 'logprobs_fname.npy')
    unitsassign_fname = os.path.join(output_directory, 'unit_assignment.npy')

    if os.path.exists(prob_fname):
        return prob_fname, outlier_fname, logprobs_fname, unit_assignment

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # reader for residual
    reader_resid = READER(residual_fname,
                          residual_dtype,
                          CONFIG,
                          .1)


    # initialize soft assignment calculator
    spatial_cov, temporal_cov = get_covariance(reader_resid, CONFIG)
    
    TAO = TEMPLATE_ASSIGN_OBJECT(
     fname_spike_train = spike_train_fname, 
     fname_templates = template_fname, 
     fname_shifts = shifts_fname,
     reader_residual = reader_resid,
     spat_cov = spatial_cov,
     temp_cov = temporal_cov,
     channel_idx = CONFIG.channel_index, 
     geom = CONFIG.geom,
     large_unit_threshold = np.inf,
     n_chans = 10,
     rec_chans = CONFIG.channel_index.shape[0], 
     sim_units = similar_units, 
     temp_thresh= similarity_threshold, 
     lik_window = window_size)
    
    replace_probs, probs, logprobs, unit_assignment = TAO.run()
    #outlier spike times/units
    cpu_sps = TAO.spike_train.cpu().numpy()
    chi2_df = (2*(window_size //2) + 1)*10
    cut_off = chi2(chi2_df).ppf(.999)
    outliers = cpu_sps[np.where(TAO.log_probs.min(1) > cut_off)[0], :]
    
    #append log_probs to spike_times
    #logprobs = np.concatenate((cpu_sps,TAO.log_probs), axis = 1)
    # compuate soft assignment
    np.save(prob_fname, replace_probs)
    np.save(outlier_fname, outliers)
    np.save(logprobs_fname, logprobs)
    np.save(unitsassign_fname, unit_assignment)
    return prob_fname, outlier_fname, logprobs_fname, unitsassign_fname
    