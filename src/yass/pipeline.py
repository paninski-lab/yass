"""
Built-in pipeline
"""
import time
import logging
import logging.config
import shutil
import os
import os.path as path
import matplotlib
matplotlib.use('Agg')

try:
    # py3
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import numpy as np
import yaml


import yass
from yass import set_config
from yass import preprocess, detect, cluster, deconvolve
from yass import read_config

from yass.util import (load_yaml, save_metadata, load_logging_config_file,
                       human_readable_time)
from yass.explore import RecordingExplorer
from yass.threshold import dimensionality_reduction as dim_red


def run(config, logger_level='INFO', clean=False, output_dir='tmp/',
        complete=False, set_zero_seed=False):
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
        Output directory (relative to CONFIG.data.root_folder to store the
        output data, defaults to tmp/

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
    logging_config['handlers']['file']['filename'] = path.join(TMP_FOLDER,
                                                               'yass.log')
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
     standardized_params,
     whiten_filter) = (preprocess
                       .run(if_file_exists=CONFIG.preprocess.if_file_exists))

    time_preprocess = time.time() - start



    ''' **********************************************
        ************** DETECT EVENTS *****************
        **********************************************
    '''
    # detect
    # Cat: This code now runs with open tensorflow calls
    start = time.time()
    (spike_index_all) = detect.run(standardized_path,
                                   standardized_params,
                                   whiten_filter,
                                   if_file_exists=CONFIG.detect.if_file_exists,
                                   save_results=CONFIG.detect.save_results)
    spike_index_clear = None
    time_detect = time.time() - start


    ''' **********************************************
        ***************** CLUSTER ********************
        **********************************************
    '''

    # cluster
    start=time.time()
    path_to_spike_train_cluster = path.join(TMP_FOLDER, 'spike_train_cluster.npy')
    if os.path.exists(path_to_spike_train_cluster)==False:
        cluster.run(spike_index_clear, spike_index_all)
    else:
        print ("\nClustering completed previously...\n\n")

    spike_train_cluster = np.load(path_to_spike_train_cluster)
    templates_cluster = np.load(os.path.join(TMP_FOLDER,'templates_cluster.npy'))    

    time_cluster = time.time()-start
    #print ("Spike train clustered: ", spike_index_cluster.shape, "spike train clear: ", 
    #        spike_train_clear.shape, " templates: ", templates.shape)


    ''' **********************************************
        ************** DECONVOLUTION *****************
        **********************************************
    '''

    # run deconvolution
    start=time.time()
    spike_train, postdeconv_templates = deconvolve.run(spike_train_cluster, 
                                                         templates_cluster,
                                                         output_directory=output_dir)
    time_deconvolution = time.time() - start

    # save spike train
    path_to_spike_train = path.join(TMP_FOLDER, 'spike_train_post_deconv_post_merge.npy')
    np.save(path_to_spike_train, spike_train)
    logger.info('Spike train saved in: {}'.format(path_to_spike_train))
    
    # save template
    path_to_templates = path.join(TMP_FOLDER, 'templates_post_deconv_post_merge.npy')
    np.save(path_to_templates, postdeconv_templates)
    logger.info('Templates saved in: {}'.format(path_to_templates))
    
    
    ''' **********************************************
        ************** POST PROCESSING****************
        **********************************************
    '''

    # save metadata in tmp
    path_to_metadata = path.join(TMP_FOLDER, 'metadata.yaml')
    logging.info('Saving metadata in {}'.format(path_to_metadata))
    save_metadata(path_to_metadata)

    # save metadata in tmp
    path_to_metadata = path.join(TMP_FOLDER, 'metadata.yaml')
    logging.info('Saving metadata in {}'.format(path_to_metadata))
    save_metadata(path_to_metadata)

    # save config.yaml copy in tmp/
    path_to_config_copy = path.join(TMP_FOLDER, 'config.yaml')

    if isinstance(config, Mapping):
        with open(path_to_config_copy, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        shutil.copy2(config, path_to_config_copy)

    logging.info('Saving copy of config: {} in {}'.format(config,
                                                          path_to_config_copy))

    # this part loads waveforms for all spikes in the spike train and scores
    # them, this data is needed to later generate phy files
    if complete:
        STANDARIZED_PATH = path.join(TMP_FOLDER, 'standardized.bin')
        PARAMS = load_yaml(path.join(TMP_FOLDER, 'standardized.yaml'))

        # load waveforms for all spikes in the spike train
        logger.info('Loading waveforms from all spikes in the spike train...')
        explorer = RecordingExplorer(STANDARIZED_PATH,
                                     spike_size=CONFIG.spike_size,
                                     dtype=PARAMS['dtype'],
                                     n_channels=PARAMS['n_channels'],
                                     data_order=PARAMS['data_order'])
        waveforms = explorer.read_waveforms(spike_train[:, 0])

        path_to_waveforms = path.join(TMP_FOLDER, 'spike_train_waveforms.npy')
        np.save(path_to_waveforms, waveforms)
        logger.info('Saved all waveforms from the spike train in {}...'
                    .format(path_to_waveforms))

        # score all waveforms
        logger.info('Scoring waveforms from all spikes in the spike train...')
        path_to_rotation = path.join(TMP_FOLDER, 'rotation.npy')
        rotation = np.load(path_to_rotation)

        main_channels = explorer.main_channel_for_waveforms(waveforms)
        path_to_main_channels = path.join(TMP_FOLDER,
                                          'waveforms_main_channel.npy')
        np.save(path_to_main_channels, main_channels)
        logger.info('Saved all waveforms main channels in {}...'
                    .format(path_to_waveforms))

        waveforms_score = dim_red.score(waveforms, rotation, main_channels,
                                        CONFIG.neigh_channels, CONFIG.geom)
        path_to_waveforms_score = path.join(TMP_FOLDER, 'waveforms_score.npy')
        np.save(path_to_waveforms_score, waveforms_score)
        logger.info('Saved all scores in {}...'.format(path_to_waveforms))

        # score templates
        # TODO: templates should be returned in the right shape to avoid .T
        templates_ = templates.T
        main_channels_tmpls = explorer.main_channel_for_waveforms(templates_)
        path_to_templates_main_c = path.join(TMP_FOLDER,
                                             'templates_main_channel.npy')
        np.save(path_to_templates_main_c, main_channels_tmpls)
        logger.info('Saved all templates main channels in {}...'
                    .format(path_to_templates_main_c))

        templates_score = dim_red.score(templates_, rotation,
                                        main_channels_tmpls,
                                        CONFIG.neigh_channels, CONFIG.geom)
        path_to_templates_score = path.join(TMP_FOLDER, 'templates_score.npy')
        np.save(path_to_templates_score, templates_score)
        logger.info('Saved all templates scores in {}...'
                    .format(path_to_waveforms))

    logger.info('Finished YASS execution. Timing summary:')
    total = (time_preprocess + time_detect + time_cluster
             + time_deconvolution)
    logger.info('\t Preprocess: %s (%.2f %%)',
                human_readable_time(time_preprocess),
                time_preprocess/total*100)
    logger.info('\t Detection: %s (%.2f %%)',
                human_readable_time(time_detect),
                time_detect/total*100)
    logger.info('\t Clustering: %s (%.2f %%)',
                human_readable_time(time_cluster),
                time_cluster/total*100)
    logger.info('\t Deconvolution: %s (%.2f %%)',
                human_readable_time(time_deconvolution),
                time_deconvolution/total*100)

    return spike_train
