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
    if os.path.exists(prob_fname):
        return prob_fname, outlier_fname

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # reader for residual
    reader_resid = READER(residual_fname,
                          residual_dtype,
                          CONFIG,
                          CONFIG.resources.n_sec_chunk_gpu_deconv/10)


    # initialize soft assignment calculator
    
    TAO = TEMPLATE_ASSIGN_OBJECT(
     fname_spike_train = spike_train_fname, 
     fname_templates = template_fname, 
     fname_shifts = shifts_fname,
     reader_residual = reader_resid,
     channel_idx = CONFIG.channel_index, 
     large_unit_threshold = np.inf,
     n_chans = 10,
     rec_chans = CONFIG.channel_index.shape[0], 
     sim_units = similar_units, 
     temp_thresh= similarity_threshold, 
     lik_window = window_size)
    
    replace_probs, probs = TAO.run()
    #outlier spike times/units
    cpu_sps = TAO.spike_train.cpu().numpy()
    chi2_df = (2*(window_size //2) + 1)*10
    cut_off = chi2(chi2_df).ppf(x = .995)
    outliers = cpu_sps[np.where(TAO.log_probs.min(1) > cut_off)[0], :]
    # compuate soft assignment
    np.save(prob_fname, replace_probs)
    np.save(outlier_fname, outliers)
    return prob_fname, outlier_fname
    