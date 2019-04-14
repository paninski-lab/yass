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


import yass
from yass import set_config
from yass import read_config
from yass import (preprocess, detect, cluster, postprocess,
                  deconvolve, residual, merge, rf, visual)

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

    ''' **********************************************
        ************** PREPROCESS ********************
        **********************************************
    '''
    # preprocess
    start = time.time()
    (standardized_path,
     standardized_params) = preprocess.run(
        os.path.join(TMP_FOLDER, 'preprocess'))

    #### Block 1: Initial run ####
    (fname_templates_init,
     fname_spike_train_init,
     fname_templates_up_init,
     fname_spike_train_up_init,
     fname_residual_init,
     residual_dtype_init) = initial_block(
        os.path.join(TMP_FOLDER, 'initial_block'),
        standardized_path,
        standardized_params)
  
    ## save the final
    fname_templates_final = os.path.join(
        TMP_FOLDER, 'templates.npy')
    fname_spike_train_final = os.path.join(
        TMP_FOLDER, 'spike_train.npy')
    # tranpose axes
    templates = np.load(fname_templates).transpose(1,2,0)
    # align spike time to the beginning
    spike_train = np.load(fname_spike_train)
    spike_train[:,0] -= CONFIG.spike_size//2
    np.save(fname_templates_final, templates)
    np.save(fname_spike_train_final, spike_train)
    
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


def initial_block(TMP_FOLDER, standardized_path, standardized_params):
    
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
        standardized_params,
        os.path.join(TMP_FOLDER, 'detect'))

    ''' **********************************************
        ***************** CLUSTER ********************
        **********************************************
    '''
    logger.info('INITIAL CLUSTERING')

    # cluster
    raw_data = True
    full_run = True
    fname_templates, fname_spike_train = cluster.run(
        spike_index_path,
        standardized_path,
        standardized_params['dtype'],
        os.path.join(TMP_FOLDER, 'cluster'),
        raw_data, 
        full_run)

    methods = ['low_ptp', 'duplicate', 'collision']
    fname_templates, fname_spike_train = postprocess.run(
        methods,
        fname_templates,
        fname_spike_train,
        os.path.join(TMP_FOLDER,
                     'cluster_post_process'),
        standardized_path,
        standardized_params['dtype'])

    ''' **********************************************
        ************** DECONVOLUTION *****************
        **********************************************
    '''

    logger.info('INITIAL DECONV')
    # run deconvolution
    (fname_templates,
     fname_spike_train,
     fname_templates_up,
     fname_spike_train_up) = deconvolve.run(
        fname_templates,
        os.path.join(TMP_FOLDER,
                     'deconv'),
        standardized_path,
        standardized_params['dtype'])

    logger.info('INITIAL RESIDUAL COMPUTATION')
    # compute residual
    fname_residual, residual_dtype = residual.run(
        fname_templates_up,
        fname_spike_train_up,
        os.path.join(TMP_FOLDER,
                     'residual'),
        standardized_path,
        standardized_params['dtype'],
        dtype_out='float32')

    return (fname_templates, fname_spike_train, fname_templates_up,
            fname_spike_train_up, fname_residual, residual_dtype)


def run_deconv_with_iterative_template_update(
    TMP_FOLDER,
    standardized_path,
    standardized_params,
    fname_templates,
    CONFIG)

    logger = logging.getLogger(__name__)

    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    update_frequency_sec = 60

    # get chunk list
    indexes = np.arange(0, CONFIG.rec_len,
        CONFIG.recordings.sampling_rate*update_frequency_sec)
    indexes = np.hstack((indexes, CONFIG.rec_len))

    idx_list = []
    for k in range(len(indexes) - 1):
        idx_list.append([indexes[k], indexes[k + 1]])
    idx_list = np.int64(np.vstack(idx_list))

    # units to updates: initially set to all
    unit_ids_recompute = None

    for chunk_idx in range(len(idx_list)):

        run_chunk =idx_list[chunk_idx]

        # run deconv
        (fname_templates,
         fname_spike_train,
         _, _) = deconvolve.run(
            fname_templates,
            os.path.join(TMP_FOLDER,
                         'deconv_chunk_{}'.format(chunk_idx)),
            standardized_path,
            standardized_params['dtype'],
            run_chunk=run_chunk,
            save_up_data=False)

         # update template
         fname_templates, diffs = update_templates(
            fname_templates,
            fname_spike_train,
            standardized_path,
            standardized_params['dtype'],
            os.path.join(TMP_FOLDER,
                'update_chunk_{}'.format(chunk_idx))
            rate=0.005,
            unit_ids=unit_ids_recompute)

         unit_ids_recompute = np.where(diffs > 0.1)[0]
         logger.info('{} units converged'.format(sum(diffs < 0.1)))







def recluster_block(TMP_FOLDER,
                    standardized_path,
                    standardized_params,
                    fname_residual,
                    residual_dtype,
                    fname_spike_train,
                    fname_templates_up,
                    fname_spike_train_up):
    
    logger = logging.getLogger(__name__)
        
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)

    ''' **********************************************
        ***************** CLUSTER ********************
        **********************************************
    '''
    # cluster
    logger.info('RECLUSTERING')
    raw_data = False
    full_run = True
    fname_templates, fname_spike_train = cluster.run(
        fname_spike_train,
        standardized_path,
        standardized_params['dtype'],
        os.path.join(TMP_FOLDER, 'cluster'),
        raw_data, 
        full_run,
        fname_residual=fname_residual,
        residual_dtype=residual_dtype,
        fname_templates_up=fname_templates_up,
        fname_spike_train_up=fname_spike_train_up)
    
    methods = ['low_ptp', 'duplicate', 'collision', 'high_mad']
    fname_templates, fname_spike_train = postprocess.run(
        methods,
        fname_templates,
        fname_spike_train,
        os.path.join(TMP_FOLDER,
                     'cluster_post_process'),
        standardized_path,
        standardized_params['dtype'])

    ''' **********************************************
        ************** DECONVOLUTION *****************
        **********************************************
    '''

    # run deconvolution
    logger.info('SECOND DECONV')
    (fname_templates,
     fname_spike_train,
     fname_templates_up,
     fname_spike_train_up) = deconvolve.run(
        fname_templates,
        os.path.join(TMP_FOLDER,
                     'deconv'),
        standardized_path,
        standardized_params['dtype'])

    # compute residual
    logger.info('SECOND RESIDUAL COMPUTATION')
    fname_residual, residual_dtype = residual.run(
        fname_templates_up,
        fname_spike_train_up,
        os.path.join(TMP_FOLDER,
                     'residual'),
        standardized_path,
        standardized_params['dtype'],
        dtype_out='float32')
    
    logger.info('FINAL MERGE')
    fname_templates, fname_spike_train = merge.run(
        os.path.join(TMP_FOLDER,
                     'post_deconv_merge'),
        False,
        fname_spike_train,
        fname_templates,
        fname_spike_train_up,
        fname_templates_up,
        standardized_path,
        standardized_params['dtype'],
        fname_residual,
        residual_dtype)

    return (fname_templates, fname_spike_train, fname_templates_up,
            fname_spike_train_up, fname_residual, residual_dtype)
