#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:44:54 2019

@author: kevin Li, JinHyung Peter Lee
"""

import os
import logging
import numpy as np
from scipy.stats import chi2

from yass import read_config
from yass.reader import READER
from yass.noise import get_noise_covariance
from yass.neuralnetwork.model_detector import Detect
from yass.noise import get_noise_covariance
from yass.soft_assignment.noise import SOFTNOISEASSIGNMENT
from yass.soft_assignment.template import TEMPLATE_ASSIGN_OBJECT

def s_score(log_probs):
    s_score = np.zeros(log_probs.shape[0])
    for i, row in enumerate(log_probs):
        col = np.argmin(row[1:3]) + 1
        s_score[i] = (row[col] - row[0])/np.max([row[0], row[col]])
    return s_score

def run(template_fname,
        spike_train_fname,
        shifts_fname,
        scales_fname,
        output_directory,
        residual_fname,
        residual_dtype,
        compute_noise_soft=True,
        compute_template_soft=True):

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    #
    fname_noise_soft = os.path.join(
        output_directory, 'noise_soft_assignment.npy')
    fname_template_soft = os.path.join(
        output_directory, 'template_soft_assignment.npz')
    
    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # reader for residual
    reader_resid = READER(residual_fname,
                          residual_dtype,
                          CONFIG,
                          CONFIG.resources.n_sec_chunk_gpu_deconv/40)

    # load NN detector
    detector = Detect(CONFIG.neuralnetwork.detect.n_filters,
                      CONFIG.spike_size_nn,
                      CONFIG.channel_index,
                      CONFIG)
    detector.load(CONFIG.neuralnetwork.detect.filename)
    detector = detector.cuda()

    ########################
    # Noise soft assignment#
    ########################

    if compute_noise_soft and (not os.path.exists(fname_noise_soft)):
        # initialize soft assignment calculator
        threshold = CONFIG.deconvolution.threshold/0.1
        sna = SOFTNOISEASSIGNMENT(spike_train_fname, template_fname, shifts_fname, scales_fname,
                                  reader_resid, detector, CONFIG.channel_index, threshold)

        # compuate soft assignment
        probs_noise = sna.compute_soft_assignment()
        np.save(fname_noise_soft, probs_noise)

    ###########################
    # Template soft assignment#
    ###########################

    if compute_template_soft and (not os.path.exists(fname_template_soft)):

        # get whitening filters
        fname_spatial_cov = os.path.join(output_directory, 'spatial_cov.npy')
        fname_temporal_cov = os.path.join(output_directory, 'temporal_cov.npy')
        if not (os.path.exists(fname_spatial_cov) and os.path.exists(fname_temporal_cov)):
            spatial_cov, temporal_cov = get_noise_covariance(reader_resid, CONFIG)
            np.save(fname_spatial_cov, spatial_cov)
            np.save(fname_temporal_cov, temporal_cov)
        else:
            spatial_cov = np.load(fname_spatial_cov)
            temporal_cov = np.load(fname_temporal_cov)
        window_size = 51
        n_chans = 10
        reader_resid = READER(residual_fname,
                      residual_dtype,
                      CONFIG,
                      CONFIG.resources.n_sec_chunk_gpu_deconv/100)

        TAO = TEMPLATE_ASSIGN_OBJECT(
            fname_spike_train = spike_train_fname, 
            fname_templates = template_fname, 
            fname_shifts = shifts_fname,
            reader_residual = reader_resid,
            spat_cov = spatial_cov,
            temp_cov = temporal_cov,
            channel_idx = CONFIG.channel_index, 
            geom = CONFIG.geom,
            large_unit_threshold = 100000,
            n_chans = n_chans,
            rec_chans = CONFIG.channel_index.shape[0], 
            sim_units = 3, 
            temp_thresh = 5, 
            lik_window = window_size)

        probs_templates, _, logprobs_outliers, units_assignment = TAO.run()
        #outlier spike times/units
        cpu_sps = TAO.spike_train.cpu().numpy()
        chi2_df = (2*(window_size //2) + 1)*n_chans
        cut_off = chi2(chi2_df).ppf(.999)
        outliers = cpu_sps[np.where(TAO.log_probs.min(1) > cut_off)[0], :]
        s_table = s_score(_)
        logprobs_outliers = logprobs_outliers/chi2_df
        #append log_probs to spike_times
        #logprobs = np.concatenate((cpu_sps,TAO.log_probs), axis = 1)
        # compuate soft assignment
        #np.save(prob_template_fname, probs_templates)
        #np.save(outlier_fname, outliers)
        #np.save(logprobs_outlier_fname, logprobs_outliers)
        #np.save(units_assign_fname, units_assignment)

        np.savez(fname_template_soft,
                 probs_templates=probs_templates,
                 units_assignment=units_assignment,
                 logprobs = _
                 sihoulette_score = s_table
                 logprobs_outliers=logprobs_outliers,
                 outliers=outliers
                )

    return fname_noise_soft, fname_template_soft
