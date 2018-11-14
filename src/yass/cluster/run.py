import logging
import datetime
import numpy as np
import os

from yass import read_config
from yass.util import file_loader, check_for_files, LoadFile
from yass.cluster.subsample import random_subsample
from yass.cluster.triage import triage
from yass.cluster.coreset import coreset
from yass.cluster.mask import getmask
from yass.cluster.util import (calculate_sparse_rhat, make_CONFIG2,
                                binary_reader, global_merge_max_dist,
                                run_cluster_features_chunks)
from yass.mfm import get_core_data
import multiprocessing as mp


#@check_for_files(
    #filenames=[
        #LoadFile('spike_train_cluster.npy'),
        #LoadFile('tmp_loc.npy'),
        #LoadFile('templates.npy')
    #],
    #mode='values',
    #relative_to=None,
    #auto_save=True)
    
def run(spike_index_clear, 
        spike_index_all,
        out_dir='tmp/',
        if_file_exists='skip',
        save_results=False):
    """Spike clustering

    Parameters
    ----------

    spike_index: numpy.ndarray (n_clear_spikes, 2), str or Path
        2D array with indexes for spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum). Or path to an npy file

    out_dir: str, optional
        Location to store/look for the generate spike train, relative to
        CONFIG.data.root_folder

    if_file_exists: str, optional
      One of 'overwrite', 'abort', 'skip'. Control de behavior for the
      spike_train_cluster.npy. file If 'overwrite' it replaces the files if
      exists, if 'abort' it raises a ValueError exception if exists,
      if 'skip' it skips the operation if the file exists (and returns the
      stored file)

    save_results: bool, optional
        Whether to save spike train to disk
        (in CONFIG.data.root_folder/relative_to/spike_train_cluster.npy),
        defaults to False

    Returns
    -------
    spike_train: (TODO add documentation)

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/cluster.py

    """
    # load files in case they are strings or Path objects
    #spike_index = file_loader(spike_index)

    CONFIG = read_config()

    startTime = datetime.datetime.now()

    Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

    logger = logging.getLogger(__name__)

    #spike_index_all = np.copy(spike_index)  # this is only spike_index clear

    # start timer
    _b = datetime.datetime.now()

    # voltage space feature clustering
    fname = os.path.join(CONFIG.data.root_folder,
                              'tmp/spike_train_cluster.npy')
    
    if os.path.exists(fname)==False:

        # check to see if 'result/' folder exists otherwise make it
        result_dir = os.path.join(CONFIG.data.root_folder,
                                  'tmp/cluster')
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        # run new voltage features-based clustering - chunk the data
        run_cluster_features_chunks(spike_index_clear, 
                                    spike_index_all, 
                                    CONFIG)
     

