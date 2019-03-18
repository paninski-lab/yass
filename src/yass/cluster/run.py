import logging
import datetime
import numpy as np
import os
from tqdm import tqdm
import parmap

from yass import read_config
from yass.util import file_loader, check_for_files, LoadFile
from yass.cluster.subsample import random_subsample
from yass.cluster.triage import triage
from yass.cluster.coreset import coreset
from yass.cluster.mask import getmask
from yass.cluster.util import (make_CONFIG2, binary_reader, 
                               global_merge_max_dist,
                               gather_clustering_result
                              )
from yass.cluster.cluster import Cluster
from yass.cluster.plot import plot_normalized_templates

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

    # check to see if 'result/' folder exists otherwise make it
    result_dir = os.path.join(CONFIG.path_to_output_directory,
                              'cluster')
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # run new voltage features-based clustering - chunk the data
    run_cluster_features_chunks(spike_index_clear, 
                                spike_index_all, 
                                CONFIG)
     


def run_cluster_features_chunks(spike_index_clear, spike_index_all,
                                CONFIG):

    ''' New voltage feature based clustering; parallel version
    ''' 
    
    # Cat: TODO: Edu said the CONFIG file can be passed as a dictionary
    CONFIG2 = make_CONFIG2(CONFIG)
    
    # loop over chunks 
    gen = 0     #Set default generation for starting clustering stpe
    assignment_global = []
    spike_index = []
    
    # parallelize over chunks of data
    #res_file = CONFIG.data.root_folder+'tmp/spike_train_cluster.npy'
    #if os.path.exists(res_file)==False: 
    
    # Cat: TODO: link spike_size in CONFIG param

    n_channels = CONFIG.recordings.n_channels
    sampling_rate = CONFIG.recordings.sampling_rate
    geometry_file = os.path.join(CONFIG.data.root_folder,
                                 CONFIG.data.geometry)

    # select length of recording to chunk data for processing;
    # Cat: TODO: read this value from CONFIG; use initial_batch_size
    n_sec_chunk = 1200
    #n_sec_chunk = 300
    
    # determine length of processing chunk based on lenght of rec
    standardized_filename = os.path.join(CONFIG.path_to_output_directory,
                                         'preprocess',
                                         'standardized.bin')
    
    fp = np.memmap(standardized_filename, dtype='float32', mode='r')
    fp_len = fp.shape[0]

    # make index list for chunk/parallel processing
    # Cat: TODO: read buffer size from CONFIG file
    # Cat: TODO: ensure all data read including residuals (there are multiple
    #                places to fix this)
    buffer_size = 200
    indexes = np.arange(0, fp_len / n_channels, sampling_rate * n_sec_chunk)
    if indexes[-1] != fp_len / n_channels:
        indexes = np.hstack((indexes, fp_len / n_channels))

    idx_list = []
    for k in range(len(indexes) - 1):
        idx_list.append([
            indexes[k], indexes[k + 1], buffer_size,
            indexes[k + 1] - indexes[k] + buffer_size
        ])

    idx_list = np.int64(np.vstack(idx_list))
    proc_indexes = np.arange(len(idx_list))

    # select first time index to make output_directories for each chunk of clustering
    chunk_index = proc_indexes[0]

    # Cat: TODO: the logic below is hardcoded for clustering a single chunk
    idx = idx_list[0]

    # read chunk of data
    print ("Clustering initial chunk: ", 
            idx[0]/CONFIG.recordings.sampling_rate, "(sec)  to  ", 
            idx[1]/CONFIG.recordings.sampling_rate, "(sec)")

    # make chunk directory if not available:
    # save chunk in own directory to enable cumulative recovery 
    chunk_dir = CONFIG.path_to_output_directory+"/cluster/chunk_"+ \
                                                str(chunk_index).zfill(6)
    if not os.path.isdir(chunk_dir):
        os.makedirs(chunk_dir)
       
    # check to see if chunk is done
    global recording_chunk
    recording_chunk = None
    
    # select which spike index to use:
    if True:
        print ("  using spike_index_all for clustering step")
        spike_index = spike_index_all.copy()
    else:
        print ("  using spike_index_clear for clustering step")
        spike_index = spike_index_clear.copy()
    
    if os.path.exists(chunk_dir+'/complete.npy')==False:
   
        # select only spike_index_clear that is in the chunk
        indexes_chunk = np.where(
                    np.logical_and(spike_index[:,0]>=idx[0], 
                    spike_index[:,0]<idx[1]))[0]
        spike_index_chunk = spike_index[indexes_chunk]
        
        # flag to indicate whether clusteirng or post-deconv reclustering
        deconv_flag = False
        full_run = False

        # Cat: TODO: this parallelization may not be optimally asynchronous
        # make arg list first
        args_in = []
        channels = np.arange(CONFIG.recordings.n_channels)
        for channel in channels:

            # check to see if chunk + channel already completed
            filename_postclustering = (chunk_dir + "/channel_"+
                                                            str(channel)+".npz")
            # skip 
            if os.path.exists(filename_postclustering):
                continue 

            args_in.append([deconv_flag, channel, CONFIG2,
                            spike_index_chunk, chunk_dir, full_run])

        print ("  starting clustering")
        if CONFIG.resources.multi_processing:
            #p = mp.Pool(CONFIG.resources.n_processors)
            #p.map_async(Cluster, args_in).get(988895)
            #p.close()
            parmap.map(Cluster, args_in, 
                       processes=CONFIG.resources.n_processors,
                       pm_pbar=True)

        else:
            with tqdm(total=len(args_in)) as pbar:
                for arg_in in args_in:
                    Cluster(arg_in)
                    pbar.update()

        ## save simple flag that chunk is done
        ## Cat: TODO: fix this; or run chunk wise-global merge
        np.save(chunk_dir+'/complete.npy',np.arange(10))
    
    else:
        print ("... clustering previously completed...")

    # Cat: TODO: this logic isn't quite correct; should merge with above
    fname = os.path.join(CONFIG.path_to_output_directory, 
                         'spike_train_cluster.npy')
    if os.path.exists(fname)==False: 

        # this flag is for deconvolution reclusters
        out_dir = 'cluster'

        # first gather clustering result
        templates, spike_train = gather_clustering_result(chunk_dir,
                                                          out_dir,
                                                          np.arange(n_channels))

        # Cat: TODO: may wish to clean up these flags; goal is to use same
        #            merge function for both clustering and deconv
        global_merge_max_dist(templates,
                              spike_train,
                              CONFIG2,
                              chunk_dir,
                              out_dir)
