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

from yass import (preprocess, augment)
from yass.neuralnetwork import Detect, Denoise

from yass.util import (load_yaml, save_metadata, load_logging_config_file,
                       human_readable_time)

def run(config, logger_level='INFO', clean=False, output_dir='tmp/'):
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

    # TODO: if input spike train is None, run yass with threshold detector
    #if fname_spike_train is None:
    #    logger.info('Not available yet. You must input spike train')
    #    return

    ''' **********************************************
        ************** PREPROCESS ********************
        **********************************************
    '''
    # preprocess
    start = time.time()
    (standardized_path,
     standardized_params) = preprocess.run(
        os.path.join(TMP_FOLDER, 'preprocess'))

    # Get training data maker
    DetectTD, DenoTD = augment.run(
        standardized_path,
        standardized_params['dtype'],
        CONFIG.neuralnetwork.training.input_spike_train_filname,
        os.path.join(TMP_FOLDER, 'augment'))

    # Train Detector
    detector = Detect(CONFIG.spike_size_small, CONFIG.channel_index)
    detector.train(CONFIG.neuralnetwork.detect.filename, DetectTD)
    
    # Train Denoiser
    deno = Denoise(CONFIG.spike_size_small)
    #deno.load('/home/peter/peter/ej49_20min/tmp/augment/test_deno.pt')
    deno.train(CONFIG.neuralnetwork.detect.filename, DenoTD)
