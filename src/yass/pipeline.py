"""
Built-in pipeline
"""
import time
import logging
import logging.config
import shutil
import os
import matplotlib
matplotlib.use('Agg')

# supress PCA unpickle userwarning 
# Cat: TODO: this is dangersous, may wish to fix the problem in cluster.py
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

try:
    # py3
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import numpy as np
import yaml
import torch
torch.multiprocessing.set_start_method('spawn', force=True)

import yass
from yass import set_config
from yass import read_config
from yass import (preprocess, detect, cluster, postprocess,
                  deconvolve, residual, soft_assignment,
                  merge, rf, visual, phy)
from yass.cluster.sharpen import sharpen_templates
from yass.reader import READER
from yass.template import run_cleaned_template_computation, run_template_computation
from yass.pd_split import run_post_deconv_split
from yass.template_update import run_template_update
from yass.deconvolve.utils import shift_svd_denoise
from yass.postprocess.duplicate_soft_assignment import duplicate_soft_assignment
from yass.soft_assignment.template import get_similar_array
#from yass.template import update_templates
from yass.deconvolve.full_template_track import full_rank_update

from yass.util import (load_yaml, save_metadata, load_logging_config_file,
                       human_readable_time)


def run(config, logger_level='INFO', clean=False, output_dir='tmp/',
        complete=False, calculate_rf=False, visualize=False, set_zero_seed=False):
            
    """Run YASS built-in pipeline

    Parameters
    ----------
    config: str or mapping (such as dictionary)
        Path to YASS configuration file or mapping object

    logger_level: str
        Logger level

    clean: bool, optional
        Delete CONFIG.data.root_folder/output_dir/ before running

    output_dir: str, optional
        Output directory (if relative, it makes it relative to
        CONFIG.data.root_folder) to store the output data, defaults to tmp/.
        If absolute, it leaves it as it is.

    complete: bool, optional
        Generates extra files (needed to generate phy files)

    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/:

    * ``config.yaml`` - Copy of the configuration file
    * ``metadata.yaml`` - Experiment metadata
    * ``filtered.bin`` - Filtered recordings (from preprocess)
    * ``filtered.yaml`` - Filtered recordings metadata (from preprocess)
    * ``standardized.bin`` - Standarized recordings (from preprocess)
    * ``standardized.yaml`` - Standarized recordings metadata (from preprocess)
    * ``whitening.npy`` - Whitening filter (from preprocess)


    Returns
    -------
    numpy.ndarray
        Spike train
    """

    # load yass configuration parameters
    set_config(config, output_dir)
    CONFIG = read_config()
    TMP_FOLDER = CONFIG.path_to_output_directory

    # remove tmp folder if needed
    if os.path.exists(TMP_FOLDER) and clean:
        shutil.rmtree(TMP_FOLDER)

    # create TMP_FOLDER if needed
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    # load logging config file
    logging_config = load_logging_config_file()
    logging_config['handlers']['file']['filename'] = os.path.join(
        TMP_FOLDER,'yass.log')
    logging_config['root']['level'] = logger_level

    # configure logging
    logging.config.dictConfig(logging_config)

    # instantiate logger
    logger = logging.getLogger(__name__)

    # print yass version
    logger.info('YASS version: %s', yass.__version__)

    ''' **********************************************
        ******** SET ENVIRONMENT VARIABLES ***********
        **********************************************
    '''
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["GIO_EXTRA_MODULES"] = "/usr/lib/x86_64-linux-gnu/gio/modules/"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)
    ''' **********************************************
        ************** PREPROCESS ********************
        **********************************************
    '''

    # preprocess
    start = time.time()
    (standardized_path,
     standardized_dtype) = preprocess.run(
        os.path.join(TMP_FOLDER, 'preprocess'))

    if CONFIG.data.initial_templates is not None:
        fname_templates = CONFIG.data.initial_templates
    else:
        #### Block 1: Detection, Clustering, Postprocess
        #print ("CLUSTERING DEFAULT LENGTH: ", CONFIG.rec_len, " current set to 300 sec")
        (fname_templates,
         fname_spike_train) = initial_block(
            os.path.join(TMP_FOLDER, 'block_1'),
            standardized_path,
            standardized_dtype,
            run_chunk_sec = CONFIG.clustering_chunk)

        print (" inpput to block2: ", fname_templates)

        #### Block 2: Deconv, Merge, Residuals, Clustering, Postprocess
        n_iterations = 1
        for it in range(n_iterations):
            (fname_templates,
             fname_spike_train) = iterative_block(
                os.path.join(TMP_FOLDER, 'block_{}'.format(it+2)),
                standardized_path,
                standardized_dtype,
                fname_templates,
                run_chunk_sec = CONFIG.clustering_chunk)

        for j in range(1):
            ### Pre-final deconv: Deconvolve, Residual, Merge, kill low fr units
            (fname_templates,
             fname_spike_train)= pre_final_deconv(
                os.path.join(TMP_FOLDER, 'pre_final_deconv_{}'.format(j)),
                standardized_path,
                standardized_dtype,
                fname_templates,
                run_chunk_sec = CONFIG.clustering_chunk)

    ### Final deconv: Deconvolve, Residual, soft assignment
    (fname_templates,
     fname_spike_train,
     fname_noise_soft, 
     fname_template_soft)= final_deconv(
        os.path.join(TMP_FOLDER, 'final_deconv'),
        standardized_path,
        standardized_dtype,
        fname_templates,
        CONFIG)

    ## save the final templates and spike train
    fname_templates_final = os.path.join(
        TMP_FOLDER, 'templates.npy')
    fname_spike_train_final = os.path.join(
        TMP_FOLDER, 'spike_train.npy')
    fname_noise_soft_assignment_final = os.path.join(
        TMP_FOLDER, 'noise_soft_assignment.npy')

    if CONFIG.deconvolution.update_templates:
        templates_dir = fname_templates
        fname_templates = os.path.join(templates_dir, 'templates_init.npy')

    # tranpose axes
    templates = np.load(fname_templates).transpose(1,2,0)
    # align spike time to the beginning
    spike_train = np.load(fname_spike_train)
    #spike_train[:,0] -= CONFIG.spike_size//2
    soft_assignment = np.load(fname_noise_soft)

    np.save(fname_templates_final, templates)
    np.save(fname_spike_train_final, spike_train)
    np.save(fname_noise_soft_assignment_final, soft_assignment)

    total_time = time.time() - start


        
    ''' **********************************************
        ************** RF / VISUALIZE ****************
        **********************************************
    '''

    if calculate_rf:
        rf.run() 

    if visualize:
        visual.run()
    
    logger.info('Finished YASS execution. Total time: {}'.format(
        human_readable_time(total_time)))
    logger.info('Final Templates Location: '+fname_templates_final)
    logger.info('Final Spike Train Location: '+fname_spike_train_final)


def initial_block(TMP_FOLDER,
                  standardized_path,
                  standardized_dtype,
                  run_chunk_sec):
    
    logger = logging.getLogger(__name__)

    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    ''' **********************************************
        ************** DETECT EVENTS *****************
        **********************************************
    '''

    # detect
    logger.info('INITIAL DETECTION')
    spike_index_path = detect.run(
        standardized_path,
        standardized_dtype,
        os.path.join(TMP_FOLDER, 'detect'),
        run_chunk_sec=run_chunk_sec)

    logger.info('INITIAL CLUSTERING')

    # cluster
    fname_templates, fname_spike_train = cluster.run(
        os.path.join(TMP_FOLDER, 'cluster'),
        standardized_path,
        standardized_dtype,
        fname_spike_index=spike_index_path,
        raw_data=True, 
        full_run=True)    

    #methods = ['duplicate', 'high_mad', 'collision']
    #methods = ['off_center', 'high_mad', 'duplicate']
    methods = ['off_center', 'high_mad', 'duplicate']
    #methods = ['off_center', 'high_mad', 'duplicate']
    (fname_templates, fname_spike_train, _, _, _) = postprocess.run(
        methods,
        os.path.join(TMP_FOLDER,
                     'cluster_post_process'),
        standardized_path,
        standardized_dtype,
        fname_templates,
        fname_spike_train)

    return fname_templates, fname_spike_train


def iterative_block(TMP_FOLDER,
                    standardized_path,
                    standardized_dtype,
                    fname_templates,
                    run_chunk_sec):

    logger = logging.getLogger(__name__)

    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    # run deconvolution
    logger.info('DECONV')
    (fname_templates,
     fname_spike_train,
     fname_shifts,
     fname_scales) = deconvolve.run(
        fname_templates,
        os.path.join(TMP_FOLDER,
                     'deconv'),
        standardized_path,
        standardized_dtype,
        run_chunk_sec=run_chunk_sec)

    # compute residual
    logger.info('RESIDUAL COMPUTATION')
    fname_residual, residual_dtype = residual.run(
        fname_shifts,
        fname_scales,
        fname_templates,
        fname_spike_train,
        os.path.join(TMP_FOLDER,
                     'residual'),
        standardized_path,
        standardized_dtype,
        dtype_out='float32',
        run_chunk_sec=run_chunk_sec)

    # cluster
    logger.info('RECLUSTERING')
    fname_templates, fname_spike_train = cluster.run(
        os.path.join(TMP_FOLDER, 'cluster'),
        standardized_path,
        standardized_dtype,
        fname_residual=fname_residual,
        residual_dtype=residual_dtype,
        fname_spike_index=None,
        fname_templates=fname_templates,
        fname_spike_train=fname_spike_train,
        fname_shifts = fname_shifts,
        fname_scales = fname_scales,
        raw_data=False, 
        full_run=True)

    methods = ['off_center', 'high_mad', 'duplicate']
    fname_templates, fname_spike_train, _, _, _ = postprocess.run(
        methods,
        os.path.join(TMP_FOLDER,
                     'cluster_post_process'),
        standardized_path,
        standardized_dtype,
        fname_templates,
        fname_spike_train)

    return fname_templates, fname_spike_train


def pre_final_deconv(TMP_FOLDER,
                     standardized_path,
                     standardized_dtype,
                     fname_templates,
                     run_chunk_sec):

    logger = logging.getLogger(__name__)

    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    ''' **********************************************
        ************** DECONVOLUTION *****************
        **********************************************
    '''

    # run deconvolution
    logger.info('DECONV')
    (fname_templates,
     fname_spike_train,
     fname_shifts,
     fname_scales) = deconvolve.run(
        fname_templates,
        os.path.join(TMP_FOLDER,
                     'deconv'),
        standardized_path,
        standardized_dtype,
        run_chunk_sec=run_chunk_sec)

    # compute residual
    logger.info('RESIDUAL COMPUTATION')
    fname_residual, residual_dtype = residual.run(
        fname_shifts,
        fname_scales,
        fname_templates,
        fname_spike_train,
        os.path.join(TMP_FOLDER,
                     'residual'),
        standardized_path,
        standardized_dtype,
        dtype_out='float32',
        run_chunk_sec=run_chunk_sec)

    logger.info('SOFT ASSIGNMENT')
    fname_noise_soft, fname_template_soft = soft_assignment.run(
        fname_templates,
        fname_spike_train,
        fname_shifts,
        fname_scales,
        os.path.join(TMP_FOLDER,
                     'soft_assignment'),
        fname_residual,
        residual_dtype,
        compute_noise_soft=True,
        compute_template_soft=True)
    
    logger.info('Remove Bad units')
    methods = ['low_fr', 'low_ptp', 'duplicate', 'duplicate_soft_assignment']
    (fname_templates, fname_spike_train, 
     fname_noise_soft, fname_shifts, fname_scales)  = postprocess.run(
        methods,
        os.path.join(TMP_FOLDER,
                     'post_deconv_post_process'),
        standardized_path,
        standardized_dtype,
        fname_templates,
        fname_spike_train,
        fname_template_soft,
        fname_noise_soft,
        fname_shifts,
        fname_scales)

    logger.info('POST DECONV MERGE')
    (fname_templates,
     fname_spike_train,
     fname_shifts,
     fname_scales,
     fname_noise_soft) = merge.run(
        os.path.join(TMP_FOLDER,
                     'post_deconv_merge'),
        fname_spike_train,
        fname_shifts,
        fname_scales,
        fname_templates,
        fname_noise_soft,
        fname_residual,
        residual_dtype)    

    #logger.info('Get (partially) Cleaned Templates')
    #fname_templates = get_partially_cleaned_templates(
    #    os.path.join(TMP_FOLDER,
    #                 'clean_templates'),
    #    fname_templates,
    #    fname_spike_train,
    #    fname_shifts,
    #    fname_scales,
    #    standardized_path,
    #    standardized_dtype,
    #    run_chunk_sec)

    return (fname_templates,
            fname_spike_train)


def get_partially_cleaned_templates(TMP_FOLDER,
                                    fname_templates,
                                    fname_spike_train,
                                    fname_shifts,
                                    fname_scales,
                                    standardized_path,
                                    standardized_dtype,
                                    run_chunk_sec):

    logger = logging.getLogger(__name__)

    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
        
    CONFIG = read_config()

    fname_templates_out = os.path.join(TMP_FOLDER, 'templates.npy')
    if os.path.exists(fname_templates_out):
        return fname_templates_out

    big_unit_ptp = 10
    vis_chan_ptp = 3
    
    # compute residual
    logger.info('PARTIAL RESIDUAL COMPUTATION')
    fname_residual, residual_dtype = residual.run(
        fname_shifts,
        fname_scales,
        fname_templates,
        fname_spike_train,
        os.path.join(TMP_FOLDER,
                     'partial_residual'),
        standardized_path,
        standardized_dtype,
        dtype_out='float32',
        run_chunk_sec=run_chunk_sec,
        min_ptp_units=big_unit_ptp,
        min_ptp_vis_chan=vis_chan_ptp)
    
    templates = np.load(fname_templates)
    n_units, n_times, n_channels = templates.shape
    ptps_mc = templates.ptp(1).max(1)

    unit_ids_big = np.where(ptps_mc > big_unit_ptp)[0]
    unit_ids_small = np.where(ptps_mc <= big_unit_ptp)[0]

    templates_input_masked = np.copy(templates)
    templates_input_masked[unit_ids_small] = 0
    for k in unit_ids_big:
        idx_noise_chan = templates_input_masked[k].ptp(0) < vis_chan_ptp
        templates_input_masked[k, :, idx_noise_chan] = 0
    fname_templates_input_masked = os.path.join(TMP_FOLDER, 'templates_input_masked.npy')
    np.save(fname_templates_input_masked, templates_input_masked)

    fname_templates_big = run_cleaned_template_computation(
        os.path.join(TMP_FOLDER,
                     'large_units_templates'),
        fname_spike_train,
        fname_templates_input_masked,
        fname_shifts,
        fname_scales,
        fname_residual,
        residual_dtype,
        CONFIG,
        unit_ids=unit_ids_big)

    reader = READER(fname_residual, residual_dtype, CONFIG)
    fname_templates_small = run_template_computation(
        os.path.join(TMP_FOLDER,
                     'small_units_templates'),
        fname_spike_train,
        reader,
        spike_size=None,
        unit_ids=unit_ids_small,
        multi_processing=CONFIG.resources.multi_processing,
        n_processors=CONFIG.resources.n_processors)
    
    
    templates_big = np.load(fname_templates_big)
    templates_small = np.load(fname_templates_small)
    templates_new = np.zeros((n_units, n_times, n_channels), 'float32')    
    templates_new[unit_ids_big] = templates_big[unit_ids_big]
    templates_new[unit_ids_small] = templates_small[unit_ids_small]

    #for unit in range(n_units):
    #    templates_new[unit,:, templates[unit].ptp(0) == 0] = 0

    #fname_templates = os.path.join(TMP_FOLDER, 'templates.npy')
    np.save(fname_templates_out, templates_new)
    
    #logger.info("subsample template alignment")
    #fname_templates_out = sharpen_templates(fname_templates,
    #                                        fname_templates_out)
    
    return fname_templates_out


def final_deconv(TMP_FOLDER,
                 standardized_path,
                 standardized_dtype,
                 fname_templates,
                 CONFIG):

    logger = logging.getLogger(__name__)

    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    ''' **********************************************
        ************** DECONVOLUTION *****************
        **********************************************
    '''

    update_templates = CONFIG.deconvolution.update_templates
    run_chunk_sec = CONFIG.final_deconv_chunk
    generate_phy = CONFIG.resources.generate_phy
        
    # run deconvolution
    logger.info('FINAL DECONV')
    if update_templates:
        (fname_templates,
         fname_spike_train,
         fname_shifts,
         fname_scales) = final_deconv_with_template_updates_v2(
            os.path.join(TMP_FOLDER,
                         'deconv_with_updates'),
            standardized_path,
            standardized_dtype,
            fname_templates,
            run_chunk_sec,
            remove_meta_data=True)
    else:
        (fname_templates,
         fname_spike_train,
         fname_shifts,
         fname_scales) = deconvolve.run(
            fname_templates,
            os.path.join(TMP_FOLDER,
                         'deconv'),
            standardized_path,
            standardized_dtype,
            run_chunk_sec=run_chunk_sec)

    # compute residual
    logger.info('RESIDUAL COMPUTATION')
    fname_residual, residual_dtype = residual.run(
        fname_shifts,
        fname_scales,
        fname_templates,
        fname_spike_train,
        os.path.join(TMP_FOLDER,
                     'residual'),
        standardized_path,
        standardized_dtype,
        dtype_out='float32',
        update_templates=update_templates,
        run_chunk_sec=run_chunk_sec)


    ''' **********************************************
        ************** GENERATE PHY FILES ************
        **********************************************
    '''
    
    if generate_phy:
        phy.run(CONFIG)
    
    logger.info('SOFT ASSIGNMENT')
    fname_noise_soft, fname_template_soft = soft_assignment.run(
        fname_templates,
        fname_spike_train,
        fname_shifts,
        fname_scales,
        os.path.join(TMP_FOLDER,
                     'soft_assignment'),
        fname_residual,
        residual_dtype,
        update_templates=update_templates)

    return (fname_templates,
            fname_spike_train,
            fname_noise_soft, 
            fname_template_soft)

def final_deconv_with_template_updates_v2(output_directory,
                                          recording_dir,
                                          recording_dtype,
                                          fname_templates_in,
                                          run_chunk_sec,
                                          remove_meta_data=True):
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    temp_directory = os.path.join(output_directory, 'templates')
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    fname_spike_train_out = os.path.join(
        output_directory, 'spike_train.npy')
    fname_shifts_out = os.path.join(
        output_directory, 'shifts.npy')
    fname_scales_out = os.path.join(
        output_directory, 'scales.npy')

    if (os.path.exists(fname_spike_train_out) and
        os.path.exists(fname_shifts_out) and
        os.path.exists(fname_scales_out)):            
        return temp_directory, fname_spike_train_out, fname_shifts_out, fname_scales_out
            
    forward_directory = os.path.join(output_directory, 'forward_results')
    if not os.path.exists(forward_directory):
        os.makedirs(forward_directory)

    CONFIG = read_config()
    template_update_freq = CONFIG.deconvolution.template_update_time
    update_time = np.arange(run_chunk_sec[0], run_chunk_sec[1], template_update_freq)
    update_time = np.hstack((update_time, run_chunk_sec[1]))

    # forward deconv
    for j in range(len(update_time)-1):

        if j == 0:
            first_batch = True
        else:
            first_batch = False

        # batch time in seconds
        batch_time = [update_time[j], update_time[j+1]]

        # templates name out
        fname_templates_out = os.path.join(
            forward_directory, 'templates_{}_{}.npy'.format(
                batch_time[0], batch_time[1]))

        output_directory_batch = os.path.join(
            forward_directory, 'batch_{}_{}'.format(batch_time[0], batch_time[1]))
        if not os.path.exists(fname_templates_out):

            # run post deconv split merge
            fname_templates_ = deconv_pass1(
                output_directory_batch,
                recording_dir,
                recording_dtype,
                fname_templates_in,
                batch_time,
                CONFIG,
                first_batch)

            # save templates and remove all metadata
            np.save(fname_templates_out, np.load(fname_templates_))

        if os.path.exists(output_directory_batch) and remove_meta_data:
            shutil.rmtree(output_directory_batch)

        # the new templates will go into the next batch as input
        fname_templates_in = fname_templates_out


    backward_directory = os.path.join(output_directory, 'backward_results')
    if not os.path.exists(backward_directory):
        os.makedirs(backward_directory)

    # adding all split units
    for j in range(len(update_time)-2, -1, -1):
        
        batch_time = [update_time[j], update_time[j+1]]
        
        fname_templates_batch = os.path.join(
            backward_directory, 'templates_{}sec.npy'.format(
                batch_time[0]))

        if os.path.exists(fname_templates_batch):
            continue

        # input templates
        fname_templates_in = os.path.join(
            forward_directory, 'templates_{}_{}.npy'.format(
                batch_time[0], batch_time[1]))

        #if False:
        if j < len(update_time)-2:

            # load current templates and templates in the next batch
            templates_current_batch = np.load(fname_templates_in)
            templates_next_batch = np.load(os.path.join(
                backward_directory, 'templates_{}sec.npy'.format(
                    update_time[j+1])))
            
            # add any new templates in the next batch
            if templates_current_batch.shape[0] < templates_next_batch.shape[0]:
                templates_current_batch = np.concatenate(
                    (templates_current_batch, templates_next_batch[templates_current_batch.shape[0]:]),
                    axis=0)
            np.save(fname_templates_batch, templates_current_batch)
                
        else:
            np.save(fname_templates_batch, np.load(fname_templates_in))

    # backward deconv

    # this makes sure that it computes the soft assignment
    # using the same units across different batches
    fname_temp = os.path.join(
        backward_directory, 'templates_{}sec.npy'.format(
            update_time[0]))
    sim_array_soft_assignment = get_similar_array(
        np.load(fname_temp), 3)

    for j in range(len(update_time)-2, -1, -1):

        batch_time = [update_time[j], update_time[j+1]]

        # all required outputs
        fname_templates_batch = os.path.join(
            backward_directory,
            'templates_{}_{}_post_update.npy'.format(batch_time[0], batch_time[1]))
        fname_spike_train_batch = os.path.join(
            backward_directory,
            'spike_train_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_shifts_batch = os.path.join(
            backward_directory,
            'shifts_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_scales_batch = os.path.join(
            backward_directory,
            'scales_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_template_soft_batch = os.path.join(
            backward_directory,
            'template_soft_{}_{}.npz'.format(batch_time[0], batch_time[1]))

        # if one of them is missing run deconv on this batch
        if (os.path.exists(fname_templates_batch) and
            os.path.exists(fname_spike_train_batch) and
            os.path.exists(fname_shifts_batch) and
            os.path.exists(fname_scales_batch)):            
            continue

        fname_templates_in = os.path.join(
            backward_directory, 'templates_{}sec.npy'.format(
                batch_time[0]))

        output_directory_batch = os.path.join(
            backward_directory, 'deconv_{}_{}'.format(
                batch_time[0], batch_time[1]))
        (fname_templates_,
         fname_spike_train_,
         fname_shifts_,
         fname_scales_,
         fname_template_soft_) = deconv_pass_2(output_directory_batch,
                                               recording_dir,
                                               recording_dtype,
                                               fname_templates_in,
                                               batch_time,
                                               CONFIG,
                                               sim_array_soft_assignment)
        np.save(fname_templates_batch, np.load(fname_templates_))
        np.save(fname_spike_train_batch, np.load(fname_spike_train_))
        np.save(fname_shifts_batch, np.load(fname_shifts_))
        np.save(fname_scales_batch, np.load(fname_scales_))

        temp_ = np.load(fname_template_soft_)
        np.savez(
            fname_template_soft_batch,
            probs_templates=temp_['probs_templates'],
            units_assignment=temp_['units_assignment'])

        if remove_meta_data:
            shutil.rmtree(output_directory_batch)
    
    # post backward process
    # gather all results and
    # kill based on soft assignment and firing rates
    units_survived = post_backward_process(backward_directory,
                                           run_chunk_sec,
                                           update_time,
                                           recording_dir,
                                           recording_dtype)
    
    # final forward pass
    final_directory = os.path.join(output_directory, 'final_pass')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # add the final templates
    for j in range(len(update_time)-1):

        batch_time = [update_time[j], update_time[j+1]]
        
        fname_templates_batch = os.path.join(
            temp_directory, 'templates_{}sec.npy'.format(
                batch_time[0]))

        if os.path.exists(fname_templates_batch):
            continue

        # input templates
        fname_templates_in = os.path.join(
            backward_directory, 'templates_{}_{}_post_update.npy'.format(
                batch_time[0], batch_time[1]))

        np.save(fname_templates_batch, np.load(fname_templates_in)[units_survived])

    for j in range(len(update_time)-1):
        
        batch_time = [update_time[j], update_time[j+1]]
        
        # all required outputs
        fname_templates_batch = os.path.join(
            temp_directory,
            'templates_{}sec.npy'.format(batch_time[0]))
        fname_spike_train_batch = os.path.join(
            final_directory,
            'spike_train_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_shifts_batch = os.path.join(
            final_directory,
            'shifts_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_scales_batch = os.path.join(
            final_directory,
            'scales_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        
        # if one of them is missing run deconv on this batch
        if (os.path.exists(fname_templates_batch) and
            os.path.exists(fname_spike_train_batch) and
            os.path.exists(fname_shifts_batch) and
            os.path.exists(fname_scales_batch)):            
            continue

        # run deconv
        output_directory_batch = os.path.join(
            final_directory, 'deconv_{}_{}'.format(batch_time[0], batch_time[1]))
        (fname_templates_,
         fname_spike_train_,
         fname_shifts_,
         fname_scales_) = deconvolve.run(
            fname_templates_batch,
            output_directory_batch,
            recording_dir,
            recording_dtype,
            run_chunk_sec=batch_time)
        
        np.save(fname_templates_batch, np.load(fname_templates_))
        np.save(fname_spike_train_batch, np.load(fname_spike_train_))
        np.save(fname_shifts_batch, np.load(fname_shifts_))
        np.save(fname_scales_batch, np.load(fname_scales_))

        # hack for now..
        if j == 0:
            fname_templates_init = os.path.join(
                temp_directory,
                'templates_init.npy')
            np.save(fname_templates_init, np.load(fname_templates_))

        if remove_meta_data:
            shutil.rmtree(output_directory_batch)


    # gather all results
    spike_train = [None]*(len(update_time)-1)
    shifts = [None]*(len(update_time)-1)
    scales = [None]*(len(update_time)-1)
    for j in range(len(update_time)-1):

        batch_time = [update_time[j], update_time[j+1]]

        # all outputs
        fname_spike_train_batch = os.path.join(
            final_directory,
            'spike_train_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_shifts_batch = os.path.join(
            final_directory,
            'shifts_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_scales_batch = os.path.join(
            final_directory,
            'scales_{}_{}.npy'.format(batch_time[0], batch_time[1]))

        spike_train[j] = np.load(fname_spike_train_batch)
        shifts[j] = np.load(fname_shifts_batch)
        scales[j] = np.load(fname_scales_batch)
    
    spike_train = np.vstack(spike_train)
    shifts = np.hstack(shifts)
    scales = np.hstack(scales)
    
    idx_sort = np.argsort(spike_train[:,0])
    spike_train = spike_train[idx_sort]
    shifts = shifts[idx_sort]
    scales = scales[idx_sort]

    np.save(fname_spike_train_out, spike_train)
    np.save(fname_shifts_out, shifts)
    np.save(fname_scales_out, scales)

    return (temp_directory, fname_spike_train_out,
            fname_shifts_out, fname_scales_out)

def final_deconv_with_template_updates(output_directory,
                                       recording_dir,
                                       recording_dtype,
                                       fname_templates_in,
                                       run_chunk_sec,
                                       remove_meta_data=True, 
                                       full_rank = True):
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
      
    temp_directory = os.path.join(output_directory, 'templates')
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    fname_spike_train_out = os.path.join(
        output_directory, 'spike_train.npy')
    fname_shifts_out = os.path.join(
        output_directory, 'shifts.npy')
    fname_scales_out = os.path.join(
        output_directory, 'scales.npy')

    if (os.path.exists(fname_spike_train_out) and
        os.path.exists(fname_shifts_out) and
        os.path.exists(fname_scales_out)):            
        return temp_directory, fname_spike_train_out, fname_shifts_out, fname_scales_out
            
    forward_directory = os.path.join(output_directory, 'forward_results')
    if not os.path.exists(forward_directory):
        os.makedirs(forward_directory)

    CONFIG = read_config()
    template_update_freq = CONFIG.deconvolution.template_update_time
    update_time = np.arange(run_chunk_sec[0], run_chunk_sec[1], template_update_freq)
    update_time = np.hstack((update_time, run_chunk_sec[1]))

    if full_rank:
        reader = READER(recording_dir,
                              recording_dtype,
                              CONFIG,
                              CONFIG.resources.n_sec_chunk_gpu_deconv)

        full_rank_track = deconvolve.full_template_track.RegressionTemplates(reader, CONFIG, forward_directory)
    else:
        full_rank_track = None

    # forward deconv
    for j in range(len(update_time)-1):

        if j == 0:
            first_batch = True
        else:
            first_batch = False

        # batch time in seconds
        batch_time = [update_time[j], update_time[j+1]]

        # templates name out
        fname_templates_out = os.path.join(
            forward_directory, 'templates_{}_{}_forward.npy'.format(
                batch_time[0], batch_time[1]))

        output_directory_batch = os.path.join(
            forward_directory, 'batch_{}_{}'.format(batch_time[0], batch_time[1]))
        if not os.path.exists(fname_templates_out):

            # run post deconv split merge
            fname_templates_ = deconv_pass1(
                output_directory_batch,
                recording_dir,
                recording_dtype,
                fname_templates_in,
                batch_time,
                CONFIG,
                first_batch, 
                full_rank_track)

            # save templates and remove all metadata
            np.save(fname_templates_out, np.load(fname_templates_))

        if os.path.exists(output_directory_batch) and remove_meta_data:
            shutil.rmtree(output_directory_batch)

        # the new templates will go into the next batch as input
        fname_templates_in = fname_templates_out


    backward_directory = os.path.join(output_directory, 'backward_results')
    if not os.path.exists(backward_directory):
        os.makedirs(backward_directory)

    # adding all split units
    for j in range(len(update_time)-2, -1, -1):
        
        batch_time = [update_time[j], update_time[j+1]]

        # all required outputs
        fname_templates_batch = os.path.join(
            temp_directory,
            'templates_{}sec.npy'.format(batch_time[0]))

        # input templates
        fname_templates_in = os.path.join(
            forward_directory, 'templates_{}_{}_forward.npy'.format(
                batch_time[0], batch_time[1]))

        #if False:
        if j < len(update_time)-2:

            # load current templates and templates in the next batch
            templates_current_batch = np.load(fname_templates_in)
            templates_next_batch = np.load(os.path.join(
                temp_directory, 'templates_{}sec.npy'.format(
                    update_time[j+1])))
            
            # add any new templates in the next batch
            if templates_current_batch.shape[0] < templates_next_batch.shape[0]:
                templates_current_batch = np.concatenate(
                    (templates_current_batch, templates_next_batch[templates_current_batch.shape[0]:]),
                    axis=0)
            np.save(fname_templates_batch, templates_current_batch)
                
        else:
            np.save(fname_templates_batch, np.load(fname_templates_in))

    # backward deconv
    for j in range(len(update_time)-2, -1, -1):
        
        batch_time = [update_time[j], update_time[j+1]]
        
        # all required outputs
        fname_templates_batch = os.path.join(
            temp_directory,
            'templates_{}sec.npy'.format(batch_time[0]))
        fname_spike_train_batch = os.path.join(
            backward_directory,
            'spike_train_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_shifts_batch = os.path.join(
            backward_directory,
            'shifts_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_scales_batch = os.path.join(
            backward_directory,
            'scales_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        
        # if one of them is missing run deconv on this batch
        if (os.path.exists(fname_templates_batch) and
            os.path.exists(fname_spike_train_batch) and
            os.path.exists(fname_shifts_batch) and
            os.path.exists(fname_scales_batch)):            
            continue

        # run deconv
        output_directory_batch = os.path.join(
            backward_directory, 'deconv_{}_{}'.format(batch_time[0], batch_time[1]))
        (fname_templates_,
         fname_spike_train_,
         fname_shifts_,
         fname_scales_) = deconvolve.run(
            fname_templates_batch,
            output_directory_batch,
            recording_dir,
            recording_dtype,
            run_chunk_sec=batch_time)
        
        np.save(fname_templates_batch, np.load(fname_templates_))
        np.save(fname_spike_train_batch, np.load(fname_spike_train_))
        np.save(fname_shifts_batch, np.load(fname_shifts_))
        np.save(fname_scales_batch, np.load(fname_scales_))

        # hack for now..
        if j == 0:
            fname_templates_init = os.path.join(
                temp_directory,
                'templates_init.npy')
            np.save(fname_templates_init, np.load(fname_templates_))

        if remove_meta_data:
            shutil.rmtree(output_directory_batch)


    # gather all results
    spike_train = [None]*(len(update_time)-1)
    shifts = [None]*(len(update_time)-1)
    scales = [None]*(len(update_time)-1)
    for j in range(len(update_time)-1):
        
        batch_time = [update_time[j], update_time[j+1]]
        
        # all outputs
        fname_spike_train_batch = os.path.join(
            backward_directory,
            'spike_train_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_shifts_batch = os.path.join(
            backward_directory,
            'shifts_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        fname_scales_batch = os.path.join(
            backward_directory,
            'scales_{}_{}.npy'.format(batch_time[0], batch_time[1]))

        spike_train[j] = np.load(fname_spike_train_batch)
        shifts[j] = np.load(fname_shifts_batch)
        scales[j] = np.load(fname_scales_batch)
    
    spike_train = np.vstack(spike_train)
    shifts = np.hstack(shifts)
    scales = np.hstack(scales)
    
    idx_sort = np.argsort(spike_train[:,0])
    spike_train = spike_train[idx_sort]
    shifts = shifts[idx_sort]
    scales = scales[idx_sort]

    np.save(fname_spike_train_out, spike_train)
    np.save(fname_shifts_out, shifts)
    np.save(fname_scales_out, scales)

    return (temp_directory, fname_spike_train_out,
            fname_shifts_out, fname_scales_out)


def post_deconv_split_merge(output_directory,
                            recording_dir,
                            recording_dtype,
                            fname_templates,
                            run_chunk_sec,
                            CONFIG,
                            first_batch=False, 
                            full_rank = None):

    # keep track of # of units in
    n_units_in = np.load(fname_templates).shape[0]

    # deconv 0
    (fname_templates,
     fname_spike_train,
     fname_shifts,
     fname_scales) = deconvolve.run(
        fname_templates,
        os.path.join(output_directory, 'deconv_0'),
        recording_dir,
        recording_dtype,
        run_chunk_sec=run_chunk_sec)
    
    # residual 0
    (fname_residual,
     residual_dtype) = residual.run(
        fname_shifts,
        fname_scales,
        fname_templates,
        fname_spike_train,
        os.path.join(output_directory, 'residual_0'),
        recording_dir,
        recording_dtype,
        dtype_out='float32',
        run_chunk_sec=run_chunk_sec)

    # post deconv split
    (fname_templates, 
     fname_spike_train,
     fname_shifts,
     fname_scales) = run_post_deconv_split(
        os.path.join(output_directory, 'pd_split_0'),
        fname_templates,
        fname_spike_train,
        fname_shifts,
        fname_scales,
        recording_dir,
        recording_dtype,
        fname_residual,
        residual_dtype,
        run_chunk_sec[0],
        first_batch)

    if first_batch:
        print("got here")
        # deconv 1
        (fname_templates,
         fname_spike_train,
         fname_shifts,
         fname_scales) = deconvolve.run(
            fname_templates,
            os.path.join(output_directory, 'deconv_1'),
            recording_dir,
            recording_dtype,
            run_chunk_sec=run_chunk_sec)
        print("got the residual")
        # residual 1
        (fname_residual,
         residual_dtype) = residual.run(
            fname_shifts,
            fname_scales,
            fname_templates,
            fname_spike_train,
            os.path.join(output_directory, 'residual_1'),
            recording_dir,
            recording_dtype,
            dtype_out='float32',
            run_chunk_sec=run_chunk_sec)
        print("got soft assignment")
        # soft assignment 1
        fname_noise_soft, fname_template_soft = soft_assignment.run(
            fname_templates,
            fname_spike_train,
            fname_shifts,
            fname_scales,
            os.path.join(output_directory,
                         'soft_assignment_1'),
            fname_residual,
            residual_dtype
            residual_dtype,
            run_chunk_sec[0])

        #logger.info('Get (partially) Cleaned Templates')
        fname_templates = get_partially_cleaned_templates(
            os.path.join(output_directory, 'clean_templates_1'),
            fname_templates,
            fname_spike_train,
            fname_shifts,
            fname_scales,
            recording_dir,
            recording_dtype,
            run_chunk_sec)
        print("partially cleaned")
        '''
        # denoise split units
        vis_threshold_strong = 1.
        vis_threshold_weak = 0.5
        rank = 5
        pad_len = int(1.5 * CONFIG.recordings.sampling_rate / 1000.)
        jitter_len = pad_len
        templates = np.load(fname_templates)
        print(templates.shape)
        templates = shift_svd_denoise(
            templates, CONFIG,
            vis_threshold_strong, vis_threshold_weak,
            rank, pad_len, jitter_len)
        np.save(fname_templates, templates)
        templates = None
        print("Denoised the templates yayyyy")
        '''
    else:
        print("passed first batch")
        if full_rank is None:
            update_weight = 100
            units_to_update = np.arange(n_units_in)
            fname_templates = run_template_update(
                os.path.join(output_directory, 'template_update_1'),
                fname_templates, fname_spike_train,
                fname_shifts, fname_scales,
                fname_residual, residual_dtype, run_chunk_sec[0],
                update_weight, units_to_update)
            
        else:
            fname_templates = full_rank_update(os.path.join(output_directory, 'template_update_1'), full_rank, run_chunk_sec, fname_spike_train, fname_templates)
        fname_noise_soft = None
        fname_template_soft = None

    # post process kill
    n_units_after_split = np.load(fname_templates).shape[0]
    if first_batch:
        units_to_process = np.arange(n_units_after_split)
        methods = ['low_fr', 'low_ptp',
                   'duplicate', 'duplicate_soft_assignment']
    else:
        units_to_process = np.arange(n_units_in, n_units_after_split)
        methods = ['low_fr', 'low_ptp', 'duplicate']
    
    (fname_templates, fname_spike_train, 
     fname_noise_soft, fname_shifts, fname_scales)  = postprocess.run(
        methods,
        os.path.join(output_directory, 'post_process_1'),
        None,
        None,
        fname_templates,
        fname_spike_train,
        fname_template_soft,
        fname_noise_soft,
        fname_shifts,
        fname_scales,
        units_to_process)
    
    n_units_out = np.load(fname_templates).shape[0]
    print('{} new units'.format(n_units_out - n_units_in))
    
    return fname_templates

def deconv_pass_2(output_directory,
                  recording_dir,
                  recording_dtype,
                  fname_templates_in,
                  run_chunk_sec,
                  CONFIG,
                  similar_array=None):

    # deconv 0
    (fname_templates,
     fname_spike_train,
     fname_shifts,
     fname_scales) = deconvolve.run(
        fname_templates_in,
        os.path.join(output_directory, 'deconv_0'),
        recording_dir,
        recording_dtype,
        run_chunk_sec=run_chunk_sec)
    # replace it to the post deconv deonised templates
    np.save(fname_templates_in, np.load(fname_templates))

    # residual 0
    (fname_residual,
     residual_dtype) = residual.run(
        fname_shifts,
        fname_scales,
        fname_templates,
        fname_spike_train,
        os.path.join(output_directory, 'residual_1'),
        recording_dir,
        recording_dtype,
        dtype_out='float32',
        run_chunk_sec=run_chunk_sec)

    # runs soft assignment
    (_, fname_template_soft) = soft_assignment.run(
        fname_templates,
        fname_spike_train,
        fname_shifts,
        fname_scales,
        os.path.join(output_directory,
                     'soft_assignment_2'),
        fname_residual,
        residual_dtype,
        run_chunk_sec[0],
        compute_noise_soft=False,
        compute_template_soft=True,
        update_templates=False,
        similar_array=similar_array)

     # run template update
    update_weight = 100
    fname_templates = run_template_update(
        os.path.join(output_directory, 'template_update_3'),
        fname_templates, fname_spike_train,
        fname_shifts, fname_scales,
        fname_residual, residual_dtype, run_chunk_sec[0],
        update_weight)
    
    return (fname_templates, fname_spike_train,
            fname_shifts, fname_scales, fname_template_soft)


def post_backward_process(backward_directory,
                          run_chunk_sec,
                          update_time,
                          recording_dir,
                          recording_dtype):
    
    fname_units_out = os.path.join(backward_directory, 'units_survived.npy')
    
    if os.path.exists(fname_units_out):
        return np.load(fname_units_out)

    # gather all results
    fname_spike_train = os.path.join(
            backward_directory, 'spike_train.npy')
    fname_shifts = os.path.join(
            backward_directory, 'shifts.npy')
    fname_scales = os.path.join(
            backward_directory, 'scales.npy')
    fname_scales = os.path.join(
            backward_directory, 'scales.npy')
    fname_template_soft = os.path.join(
            backward_directory, 'template_soft.npz')
    if not (os.path.exists(fname_spike_train) and
            os.path.exists(fname_shifts) and
            os.path.exists(fname_scales) and
            os.path.exists(fname_template_soft)):

        spike_train = [None]*(len(update_time)-1)
        shifts = [None]*(len(update_time)-1)
        scales = [None]*(len(update_time)-1)
        probs_templates = [None]*(len(update_time)-1)
        units_assignment = [None]*(len(update_time)-1)
        for j in range(len(update_time)-1):

            batch_time = [update_time[j], update_time[j+1]]

            # all outputs
            fname_spike_train_batch = os.path.join(
                backward_directory,
                'spike_train_{}_{}.npy'.format(batch_time[0], batch_time[1]))
            fname_shifts_batch = os.path.join(
                backward_directory,
                'shifts_{}_{}.npy'.format(batch_time[0], batch_time[1]))
            fname_scales_batch = os.path.join(
                backward_directory,
                'scales_{}_{}.npy'.format(batch_time[0], batch_time[1]))
            fname_template_soft_batch = os.path.join(
                backward_directory,
                'template_soft_{}_{}.npz'.format(batch_time[0], batch_time[1]))

            spike_train[j] = np.load(fname_spike_train_batch)
            shifts[j] = np.load(fname_shifts_batch)
            scales[j] = np.load(fname_scales_batch)

            temp_ = np.load(fname_template_soft_batch)
            probs_templates[j] = temp_['probs_templates']
            units_assignment[j] = temp_['units_assignment']

        spike_train = np.vstack(spike_train)
        shifts = np.hstack(shifts)
        scales = np.hstack(scales)
        probs_templates = np.vstack(probs_templates)
        units_assignment = np.vstack(units_assignment)

        idx_sort = np.argsort(spike_train[:,0])
        spike_train = spike_train[idx_sort]
        shifts = shifts[idx_sort]
        scales = scales[idx_sort]
        probs_templates = probs_templates[idx_sort]
        units_assignment = units_assignment[idx_sort]

        np.save(fname_spike_train, spike_train)
        np.save(fname_shifts, shifts)
        np.save(fname_scales, scales)
        np.savez(
            fname_template_soft,
            probs_templates=probs_templates,
            units_assignment=units_assignment)
        
    ## residual
    #fname_residual, residual_dtype = residual.run(
    #    fname_shifts,
    #    fname_scales,
    #    backward_directory,
    #    fname_spike_train,
    #    os.path.join(backward_directory,
    #                 'post_backward_residual'),
    #    recording_dir,
    #    recording_dtype,
    #    dtype_out='float32',
    #    update_templates=True,
    #    run_chunk_sec=run_chunk_sec)

    ## soft assignment
    #_, fname_template_soft = soft_assignment.run(
    #    backward_directory,
    #    fname_spike_train,
    #    fname_shifts,
    #    fname_scales,
    #    os.path.join(backward_directory,
    #                 'post_backward_soft_assignment'),
    #    fname_residual,
    #    residual_dtype,
    #    run_chunk_sec[0],
    #    compute_noise_soft=False,
    #    compute_template_soft=True,
    #    update_templates=True)
    
    # kill units
    spike_train = np.load(fname_spike_train)
    n_units = int(np.max(spike_train[:,1]) + 1)

    units_in = np.arange(n_units)

    # kill low firing rates units
    n_spikes = np.zeros((len(update_time)-1, n_units), 'int32')
    for j in range(len(update_time)-1):

        batch_time = [update_time[j], update_time[j+1]]
        fname_spike_train_batch = os.path.join(
            backward_directory,
            'spike_train_{}_{}.npy'.format(batch_time[0], batch_time[1]))
        spike_train = np.load(fname_spike_train_batch)

        unique_units, n_counts = np.unique(spike_train[:, 1], return_counts=True)
        n_spikes[j, unique_units] = n_counts
    max_n_spikes = np.max(n_spikes, 0)
    units_in = units_in[max_n_spikes/(update_time[1] - update_time[0]) > 0.2]

    # post process kill
    units_out = duplicate_soft_assignment(fname_template_soft,
                                          units_in=units_in)

    np.save(fname_units_out, units_out)

    return units_out


    