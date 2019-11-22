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
                  deconvolve, residual, noise, merge, rf, visual, phy)
#from yass.template import update_templates

from yass.util import (load_yaml, save_metadata, load_logging_config_file,
                       human_readable_time)


def run(config, logger_level='INFO', clean=False, output_dir='tmp/',
        complete=False, calculate_rf=False, visualize=False, set_zero_seed=False,
        generate_phy=True):
            
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

    ''' **********************************************
        ************** PREPROCESS ********************
        **********************************************
    '''

    # preprocess
    start = time.time()
    (standardized_path,
     standardized_dtype) = preprocess.run(
        os.path.join(TMP_FOLDER, 'preprocess'))

    #### Block 1: Detection, Clustering, Postprocess
    #print ("CLUSTERING DEFAULT LENGTH: ", CONFIG.rec_len, " current set to 300 sec")
    (fname_templates,
     fname_spike_train) = initial_block(
        os.path.join(TMP_FOLDER, 'block_1'),
        standardized_path,
        standardized_dtype,
        run_chunk_sec = CONFIG.clustering_chunk)
        #run_chunk_sec = [0, 600*20000])
        #run_chunk_sec = [0, 300])

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

    ### Pre-final deconv: Deconvolve, Residual, Merge, kill low fr units
    (fname_templates,
     fname_spike_train)= pre_final_deconv(
        os.path.join(TMP_FOLDER, 'pre_final_deconv'),
        standardized_path,
        standardized_dtype,
        fname_templates,
        run_chunk_sec = CONFIG.clustering_chunk)
    
    ### Final deconv: Deconvolve, Residual, soft assignment
    (fname_templates,
     fname_spike_train,
     fname_soft_assignment)= final_deconv(
        os.path.join(TMP_FOLDER, 'final_deconv'),
        standardized_path,
        standardized_dtype,
        fname_templates,
        update_templates = CONFIG.deconvolution.update_templates,
        run_chunk_sec = CONFIG.final_deconv_chunk)

    ## save the final templates and spike train
    fname_templates_final = os.path.join(
        TMP_FOLDER, 'templates.npy')
    fname_spike_train_final = os.path.join(
        TMP_FOLDER, 'spike_train.npy')
    fname_soft_assignment_final = os.path.join(
        TMP_FOLDER, 'soft_assignment.npy')

    # tranpose axes
    templates = np.load(fname_templates).transpose(1,2,0)
    # align spike time to the beginning
    spike_train = np.load(fname_spike_train)
    #spike_train[:,0] -= CONFIG.spike_size//2
    soft_assignment = np.load(fname_soft_assignment)
    np.save(fname_templates_final, templates)
    np.save(fname_spike_train_final, spike_train)
    np.save(fname_soft_assignment_final, soft_assignment)

    total_time = time.time() - start

    ''' **********************************************
        ************** GENERATE PHY FILES ************
        **********************************************
    '''
    
    if generate_phy:
        phy.run(CONFIG)
    
        
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
    fname_templates, fname_spike_train = postprocess.run(
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
     fname_templates_up,
     fname_spike_train_up, 
     fname_shifts) = deconvolve.run(
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
        fname_templates,
        fname_spike_train,
        os.path.join(TMP_FOLDER,
                     'residual'),
        standardized_path,
        standardized_dtype,
        dtype_out='float32',
        run_chunk_sec=run_chunk_sec)

    #logger.info('KILL NOISE')
    #fname_spike_train2 = noise.run(
    #    fname_templates,
    #    fname_spike_train,
    #    fname_shifts,
    #    os.path.join(TMP_FOLDER,
    #                 'noise_kill'),
    #    fname_residual,
    #    residual_dtype)

    if False:
        
        logger.info('SOFT NOISE ASSIGNMENT')
        fname_soft_assignment = noise.run(
            fname_templates,
            fname_spike_train,
            fname_shifts,
            os.path.join(TMP_FOLDER,
                         'soft_assignment'),
            fname_residual,
            residual_dtype)
    
        logger.info('BLOCK1 MERGE')
        _, _, _ = merge.run(
            os.path.join(TMP_FOLDER,
                         'post_deconv_merge'),
            fname_spike_train,
            fname_templates,
            fname_soft_assignment,
            fname_residual,
            residual_dtype)

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
        raw_data=False, 
        full_run=True)

    methods = ['off_center', 'high_mad', 'duplicate']
    fname_templates, fname_spike_train = postprocess.run(
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
     fname_templates_up,
     fname_spike_train_up,
     fname_shifts) = deconvolve.run(
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
        fname_templates,
        fname_spike_train,
        os.path.join(TMP_FOLDER,
                     'residual'),
        standardized_path,
        standardized_dtype,
        dtype_out='float32',
        run_chunk_sec=run_chunk_sec)

    logger.info('SOFT NOISE ASSIGNMENT')
    fname_soft_assignment = noise.run(
        fname_templates,
        fname_spike_train,
        fname_shifts,
        os.path.join(TMP_FOLDER,
                     'soft_assignment'),
        fname_residual,
        residual_dtype)

    logger.info('POST DECONV MERGE')
    (fname_templates,
     fname_spike_train,
     fname_soft_assignment) = merge.run(
        os.path.join(TMP_FOLDER,
                     'post_deconv_merge'),
        fname_spike_train,
        fname_shifts,
        fname_templates,
        fname_soft_assignment,
        fname_residual,
        residual_dtype)
    
    logger.info('Remove Low Firing Rate Units')
    methods = ['low_fr']
    fname_templates, fname_spike_train = postprocess.run(
        methods,
        os.path.join(TMP_FOLDER,
                     'post_deconv_post_process'),
        standardized_path,
        standardized_dtype,
        fname_templates,
        fname_spike_train,
        fname_soft_assignment)

    return (fname_templates,
            fname_spike_train)


def final_deconv(TMP_FOLDER,
                 standardized_path,
                 standardized_dtype,
                 fname_templates,
                 update_templates,
                 run_chunk_sec):

    logger = logging.getLogger(__name__)

    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    ''' **********************************************
        ************** DECONVOLUTION *****************
        **********************************************
    '''

    # run deconvolution
    logger.info('FINAL DECONV')
    (fname_templates,
     fname_spike_train,
     fname_templates_up,
     fname_spike_train_up,
     fname_shifts) = deconvolve.run(
        fname_templates,
        os.path.join(TMP_FOLDER,
                     'deconv'),
        standardized_path,
        standardized_dtype,
        update_templates=update_templates,
        run_chunk_sec=run_chunk_sec)

    # compute residual
    logger.info('RESIDUAL COMPUTATION')
    fname_residual, residual_dtype = residual.run(
        fname_shifts,
        fname_templates,
        fname_spike_train,
        os.path.join(TMP_FOLDER,
                     'residual'),
        standardized_path,
        standardized_dtype,
        dtype_out='float32',
        update_templates=update_templates,
        run_chunk_sec=run_chunk_sec)

    logger.info('SOFT NOISE ASSIGNMENT')
    fname_soft_assignment = noise.run(
        fname_templates,
        fname_spike_train,
        fname_shifts,
        os.path.join(TMP_FOLDER,
                     'soft_assignment'),
        fname_residual,
        residual_dtype)

    return (fname_templates,
            fname_spike_train,
            fname_soft_assignment)
