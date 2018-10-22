"""
Detection pipeline
"""
import logging
import os.path
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path
from functools import reduce
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import parmap
import numpy as np

from yass import read_config, GPU_ENABLED
from yass.batch import BatchProcessor, RecordingsReader
from yass.threshold.detect import threshold
from yass.threshold import detect
from yass.threshold.dimensionality_reduction import pca
from yass import neuralnetwork
from yass.neuralnetwork import NeuralNetDetector, AutoEncoder, KerasModel
from yass.preprocess import whiten
from yass.geometry import n_steps_neigh_channels
from yass.util import file_loader, save_numpy_object
from keras import backend as K
from collections import defaultdict

from yass.templates.util import strongly_connected_components_iterative


def run(standarized_path, standarized_params,
        whiten_filter, output_directory='tmp/',
        if_file_exists='skip', save_results=False):

            
    """Execute detect step

    Cat: THIS CODE KEEP TENSORFLOW OPEN FOR DETECTION AND THEN COMPUTES 
         corrections post-detection

    Parameters
    ----------
    standarized_path: str or pathlib.Path
        Path to standarized data binary file

    standarized_params: dict, str or pathlib.Path
        Dictionary with standarized data parameters or path to a yaml file

    channel_index: numpy.ndarray, str or pathlib.Path
        Channel index or path to a npy file

    whiten_filter: numpy.ndarray, str or pathlib.Path
        Whiten matrix or path to a npy file

    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/

    if_file_exists: str, optional
      One of 'overwrite', 'abort', 'skip'. Control de behavior for every
      generated file. If 'overwrite' it replaces the files if any exist,
      if 'abort' it raises a ValueError exception if any file exists,
      if 'skip' if skips the operation if any file exists

    save_results: bool, optional
        Whether to save results to disk, defaults to False

    Returns
    -------
    clear_scores: numpy.ndarray (n_spikes, n_features, n_channels)
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels

    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    spike_index_call: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for all spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)

    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/ (if save_results is
    True):

    * ``spike_index_clear.npy`` - Same as spike_index_clear returned
    * ``spike_index_all.npy`` - Same as spike_index_collision returned
    * ``rotation.npy`` - Rotation matrix for dimensionality reduction
    * ``scores_clear.npy`` - Scores for clear spikes

    Threshold detector runs on CPU, neural network detector runs CPU and GPU,
    depending on how tensorflow is configured.

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/detect.py
    """
    CONFIG = read_config()

    # load files in case they are strings or Path objects
    standarized_params = file_loader(standarized_params)
    whiten_filter = file_loader(whiten_filter)

    # run detection
    if CONFIG.detect.method == 'threshold':
        return run_threshold(standarized_path,
                             standarized_params,
                             whiten_filter,
                             output_directory,
                             if_file_exists,
                             save_results)
    elif CONFIG.detect.method == 'nn':
        return run_neural_network(standarized_path,
                                  standarized_params,
                                  whiten_filter,
                                  output_directory,
                                  if_file_exists,
                                  save_results)


def run_threshold(standarized_path, standarized_params,
                  whiten_filter, output_directory, if_file_exists,
                  save_results):
    """Run threshold detector and dimensionality reduction using PCA


    Returns
    -------
    scores
      Scores for all spikes

    spike_index_clear
      Spike indexes for clear spikes

    spike_index_all
      Spike indexes for all spikes
    """
    logger = logging.getLogger(__name__)

    logger.debug('Running threshold detector...')

    CONFIG = read_config()

    if os.path.isabs(output_directory):
        folder = Path(output_directory)
    else:
        folder = Path(CONFIG.path_to_output_directory, 'detect')

    folder.mkdir(exist_ok=True)
    TMP_FOLDER = str(folder)

    # files that will be saved if enable by the if_file_exists option
    filename_index_clear = 'spike_index_clear.npy'
    filename_index_clear_pca = 'spike_index_clear_pca.npy'
    filename_scores_clear = 'scores_clear.npy'
    filename_spike_index_all = 'spike_index_all.npy'
    filename_rotation = 'rotation.npy'

    ###################
    # Spike detection #
    ###################

    # run threshold detection, save clear indexes in TMP/filename_index_clear
    clear = threshold(standarized_path,
                      standarized_params['dtype'],
                      standarized_params['n_channels'],
                      standarized_params['data_order'],
                      CONFIG.resources.max_memory,
                      CONFIG.neigh_channels,
                      CONFIG.spike_size,
                      CONFIG.spike_size + CONFIG.templates.max_shift,
                      CONFIG.detect.threshold_detector.std_factor,
                      TMP_FOLDER,
                      spike_index_clear_filename=filename_index_clear,
                      if_file_exists=if_file_exists)

    #######
    # PCA #
    #######

    # run PCA, save rotation matrix and pca scores under TMP_FOLDER
    # TODO: remove clear as input for PCA and create an independent function
    pca_scores, clear, _ = pca(standarized_path,
                               standarized_params['dtype'],
                               standarized_params['n_channels'],
                               standarized_params['data_order'],
                               clear,
                               CONFIG.spike_size,
                               CONFIG.detect.temporal_features,
                               CONFIG.neigh_channels,
                               CONFIG.channel_index,
                               CONFIG.resources.max_memory,
                               TMP_FOLDER,
                               'scores_pca.npy',
                               filename_rotation,
                               filename_index_clear_pca,
                               if_file_exists)

    #################
    # Whiten scores #
    #################

    # apply whitening to scores
    scores = whiten.score(pca_scores, clear[:, 1], whiten_filter)

    if TMP_FOLDER and save_results:
        # saves whiten scores
        path_to_scores = os.path.join(TMP_FOLDER, filename_scores_clear)
        save_numpy_object(scores, path_to_scores, if_file_exists,
                          name='scores')

        # save spike_index_all (same as spike_index_clear for threshold
        # detector)
        path_to_spike_index_all = os.path.join(TMP_FOLDER,
                                               filename_spike_index_all)
        save_numpy_object(clear, path_to_spike_index_all, if_file_exists,
                          name='Spike index all')

    return clear, np.copy(clear)


def run_neural_network(standarized_path, standarized_params,
                       whiten_filter, output_directory, if_file_exists,
                       save_results):
                           
    """Run neural network detection and autoencoder dimensionality reduction

    Returns
    -------
    scores
      Scores for all spikes

    spike_index_clear
      Spike indexes for clear spikes

    spike_index_all
      Spike indexes for all spikes
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    TMP_FOLDER = CONFIG.path_to_output_directory

    # check if all scores, clear and collision spikes exist..
    path_to_score = os.path.join(TMP_FOLDER, 'scores_clear.npy')
    path_to_spike_index_clear = os.path.join(TMP_FOLDER,
                                             'spike_index_clear.npy')
    path_to_spike_index_all = os.path.join(TMP_FOLDER, 'spike_index_all.npy')
    path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')

    path_to_standardized = os.path.join(TMP_FOLDER, 'standarized.bin')

    paths = [path_to_score, path_to_spike_index_clear, path_to_spike_index_all]
    exists = [os.path.exists(p) for p in paths]

    if (if_file_exists == 'overwrite' or not any(exists)):

        max_memory = (CONFIG.resources.max_memory_gpu if GPU_ENABLED else
                      CONFIG.resources.max_memory)

        # make tensorflow tensors and neural net classes
        detection_th = CONFIG.detect.neural_network_detector.threshold_spike
        triage_th = CONFIG.detect.neural_network_triage.threshold_collision
        detection_fname = CONFIG.detect.neural_network_detector.filename
        ae_fname = CONFIG.detect.neural_network_autoencoder.filename
        triage_fname = CONFIG.detect.neural_network_triage.filename
        n_channels = CONFIG.recordings.n_channels
      
        # open tensorflow for every chunk
        NND = NeuralNetDetector.load(detection_fname, detection_th,
                                     CONFIG.channel_index)
        NNAE = AutoEncoder.load(ae_fname, input_tensor=NND.waveform_tf)

        # run nn preprocess batch-wsie
        neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)
        
        # compute len of recording
        filename_dat = os.path.join(CONFIG.data.root_folder,
                                    CONFIG.data.recordings)
        fp = np.memmap(filename_dat, dtype='int16', mode='r')
        fp_len = fp.shape[0] / n_channels

        # compute batch indexes
        buffer_size = 200       # Cat: to set this in CONFIG file
        sampling_rate = CONFIG.recordings.sampling_rate
        #n_sec_chunk = CONFIG.resources.n_sec_chunk
        
        # Cat: TODO: Set a different size chunk for clustering vs. detection
        n_sec_chunk = 60
        
        # take chunks
        indexes = np.arange(0, fp_len, sampling_rate * n_sec_chunk)
        
        # add last bit of recording if it's shorter
        if indexes[-1] != fp_len :
            indexes = np.hstack((indexes, fp_len ))

        idx_list = []
        for k in range(len(indexes) - 1):
            idx_list.append([
                indexes[k], indexes[k + 1], buffer_size,
                indexes[k + 1] - indexes[k] + buffer_size
            ])

        idx_list = np.int64(np.vstack(idx_list))

        idx_list = idx_list
        
        logger.info("# of chunks: %i", len(idx_list))
        
        logger.info (idx_list)
        
        # run tensorflow 
        processing_ctr = 0
        #chunk_ctr = 0
        
        # chunks to cycle over are 10 x as much as initial chosen data
        total_processing = len(idx_list)*n_sec_chunk

        # keep tensorflow open
        # save iteratively
        fname_detection = os.path.join(CONFIG.path_to_output_directory,
                                       'detect')
        if not os.path.exists(fname_detection):
            os.mkdir(fname_detection)
        
        # set tensorflow verbosity level
        tf.logging.set_verbosity(tf.logging.ERROR)

        # open etsnrflow session
        with tf.Session() as sess:
            #K.set_session(sess)
            NND.restore(sess)

            #triage = KerasModel(triage_fname,
            #                    allow_longer_waveform_length=True,
            #                    allow_more_channels=True)

            # read chunks of data first:
            # read chunk of raw standardized data
            # Cat: TODO: don't save to lists, might want to use numpy arrays directl
            #print (os.path.join(fname_detection,"detect_"+
            #                      str(chunk_ctr).zfill(5)+'.npz'))

            # loop over 10sec or 60 sec chunks
            for chunk_ctr, idx in enumerate(idx_list): 
                if os.path.exists(os.path.join(fname_detection,"detect_"+
                                  str(chunk_ctr).zfill(5)+'.npz')):
                                           continue

                # reset lists 
                spike_index_list = []
                #idx_clean_list = []
                energy_list = []
                TC_list = []
                offset_list = []

                # load chunk of data
                standardized_recording = binary_reader(idx, buffer_size, path_to_standardized,
                                          n_channels, CONFIG.data.root_folder)
                
                # run detection on smaller chunks of data, e.g. 1 sec
                # Cat: TODO: add last bit at end in case short
                indexes = np.arange(0, standardized_recording.shape[0], int(sampling_rate/2))

                # run tensorflow over 1sec chunks in general
                for ctr, index in enumerate(indexes[:-1]): 
                    
                    # save absolute index of each subchunk
                    offset_list.append(idx[0]+indexes[ctr])
                    
                    data_temp = standardized_recording[indexes[ctr]:indexes[ctr+1]]
    
                    # store size of recordings in case at end of dataset.
                    TC_list.append(data_temp.shape)
                    
                    # run detect nn
                    res = NND.predict_recording(data_temp, sess=sess,
                                                output_names=('spike_index',
                                                              'waveform'))
                    spike_index, wfs = res

                    #idx_clean = (triage
                    #             .predict_with_threshold(x=wfs,
                    #                                     threshold=triage_th))

                    score = NNAE.predict(wfs, sess)
                    rot = NNAE.load_rotation(sess)
                    neighbors = n_steps_neigh_channels(CONFIG.neigh_channels,
                                                       2)

                    # idx_clean is the indexes of clear spikes in all_spikes
                    spike_index_list.append(spike_index)
                    #idx_clean_list.append(idx_clean)

                    # run AE nn; required for remove_axon function
                    # Cat: TODO: Do we really need this: can we get energy list faster?
                    #rot = NNAE.load_rotation()
                    energy_ = np.ptp(np.matmul(score[:, :, 0], rot.T), axis=1)
                    energy_list.append(energy_)
                   
                    logger.info('processed chunk: %s/%s,  # spikes: %s', 
                      str(processing_ctr), str(total_processing), spike_index.shape)
            
                    processing_ctr+=1
            
                # save chunk of data in case crashes occur
                np.savez(os.path.join(fname_detection,"detect_"+
                                       str(chunk_ctr).zfill(5)),
                                       spike_index_list = spike_index_list,
                                       energy_list = energy_list, 
                                       TC_list = TC_list,
                                       offset_list = offset_list)
        
        # load all saved data;
        spike_index_list = []
        energy_list = []
        TC_list = []
        offset_list = []
        for ctr, idx in enumerate(idx_list): 
            data = np.load(fname_detection+'/detect_'+str(ctr).zfill(5)+'.npz')
            spike_index_list.extend(data['spike_index_list'])
            energy_list.extend(data['energy_list'])
            TC_list.extend(data['TC_list'])
            offset_list.extend(data['offset_list'])

        # save all detected spikes pre axon_kill
        np.save(os.path.join(TMP_FOLDER,'spike_index_all_pre_deduplication.npy'), 
                             spike_index_list)
        
        
        # remove axons - compute axons to be killed
        logger.info(' removing axons in parallel')
        multi_procesing = 1
        if CONFIG.resources.multi_processing:
            keep = parmap.map(deduplicate,
                                list(zip(spike_index_list, 
                                energy_list, 
                                TC_list,
                                np.arange(len(energy_list)))),
                                neighbors,
                                processes=CONFIG.resources.n_processors,
                                pm_pbar=True)
        else:
            keep=[]
            for k in range(len(energy_list)):
                keep.append(deduplicate((spike_index_list[k], 
                                                     energy_list[k], 
                                                     TC_list[k],k),
                                                     neighbors))


        # Cat: TODO Note that we're killing spike_index_all as well.
        # remove axons from clear spikes - keep only non-killed+clean events
        spike_index_all_postkill = []
        for k in range(len(spike_index_list)):
            spike_index_all_postkill.append(spike_index_list[k][keep[k][0]])

        logger.info(' fixing indexes from batching')
        spike_index_all = fix_indexes_firstbatch_3(
                spike_index_all_postkill, 
                offset_list,
                buffer_size, 
                sampling_rate)
                
        # get and clean all spikes
        spikes_all = spike_index_all 
        
        #logger.info('Removing all indexes outside the allowed range to '
        #            'draw a complete waveform...')
        _n_observations = fp_len
        spikes_all, _ = detect.remove_incomplete_waveforms(
            spikes_all, CONFIG.spike_size + CONFIG.templates.max_shift,
            _n_observations)

        np.save(os.path.join(TMP_FOLDER,'spike_index_all.npy'),spikes_all)
    
    else:
        spikes_all = np.load(os.path.join(TMP_FOLDER,'spike_index_all.npy'))

    return spikes_all

def binary_reader(idx_list, buffer_size, standardized_filename,
                  n_channels, root_folder):

    # prPurple("Processing chunk: "+str(chunk_idx))

    # New indexes
    idx_start = idx_list[0]
    idx_stop = idx_list[1]
    idx_local = idx_list[2]

    data_start = idx_start
    data_end = idx_stop
    offset = idx_local

    # ***** LOAD RAW RECORDING *****
    with open(standardized_filename, "rb") as fin:
        if data_start == 0:
            # Seek position and read N bytes
            recordings_1D = np.fromfile(
                fin,
                dtype='float32',
                count=(data_end + buffer_size) * n_channels)
            recordings_1D = np.hstack((np.zeros(
                buffer_size * n_channels, dtype='float32'), recordings_1D))
        else:
            fin.seek((data_start - buffer_size) * 4 * n_channels, os.SEEK_SET)
            recordings_1D = np.fromfile(
                fin,
                dtype='float32',
                count=((data_end - data_start + buffer_size * 2) * n_channels))

        if len(recordings_1D) != (
              (data_end - data_start + buffer_size * 2) * n_channels):
            recordings_1D = np.hstack((recordings_1D,
                                       np.zeros(
                                           buffer_size * n_channels,
                                           dtype='float32')))

    fin.close()

    # Convert to 2D array
    recording = recordings_1D.reshape(-1, n_channels)
    
    return recording

                              
def deduplicate(data_in, neighbors):
    
    # default window for deduplication in timesteps
    # Cat: TODO: read from CONFIG file
    w=5
    
    # print ("removing axons: ", data_in[0].shape)
    (spike_index, energy, idx) = data_in[0], data_in[1], data_in[3]
    #(T,C) = data_in[2][0], data_in[2][1]
    #print (spike_index.shape)
    
            
    # number of data points
    n_data = spike_index.shape[0]

    # separate time and channel info
    TT = spike_index[:, 0]
    CC = spike_index[:, 1]

    # Index counting
    # indices in index_counter[t-1]+1 to index_counter[t] have time t
    T_max = np.max(TT)
    T_min = np.min(TT)
    index_counter = np.zeros(T_max + w + 1, 'int32')
    t_now = T_min
    for j in range(n_data):
        if TT[j] > t_now:
            index_counter[t_now:TT[j]] = j - 1
            t_now = TT[j]
    index_counter[T_max:] = n_data - 1

    # connecting edges
    j_now = 0
    edges = defaultdict(list)
    for t in range(T_min, T_max + 1):

        # time of j_now to index_counter[t] is t
        # j_now to index_counter[t+w] has all index from t to t+w
        max_index = index_counter[t+w]
        cc_temporal_neighs = CC[j_now:max_index+1]

        for j in range(index_counter[t]-j_now+1):
            # check if channels are also neighboring
            idx_neighs = np.where(
                neighbors[cc_temporal_neighs[j],
                          cc_temporal_neighs[j+1:]])[0] + j + 1 + j_now

            # connect edges to neighboring spikes
            for j2 in idx_neighs:
                edges[j2].append(j+j_now)
                edges[j+j_now].append(j2)

        # update j_now
        j_now = index_counter[t]+1

    # Using scc, build connected components from the graph
    idx_survive = np.zeros(n_data, 'bool')
    for scc in strongly_connected_components_iterative(np.arange(n_data),
                                                       edges):
        idx = list(scc)
        idx_survive[idx[np.argmax(energy[idx])]] = 1

    return (idx_survive, idx)
    
    
    
def remove_axons_parallel(data_in, neighbors):

    # print ("removing axons: ", data_in[0].shape)
    (spike_index, energy, idx) = data_in[0], data_in[1], data_in[3]
    (T,C) = data_in[2][0], data_in[2][1]

    # parallelize over spike_index_list chunks
    n_data = spike_index.shape[0]

    temp = np.ones((T, C), 'int32')*-1
    temp[spike_index[:, 0], spike_index[:, 1]] = np.arange(n_data)

    kill = np.zeros(n_data, 'bool')
    energy_killed = np.zeros(n_data, 'float32')
    search_order = np.argsort(energy)[::-1]

    # loop over all spikes
    for j in range(n_data):
        kill, energy_killed = kill_spikes(temp, neighbors, spike_index,
                                          energy, kill,
                                          energy_killed, search_order[j])

    # return killed data but also index of chunk to fix indexes
    #print ("remaining non-killed: ", kill.sum())
    return (kill, idx)


def kill_spikes(temp, neighbors, spike_index, energy, kill,
                energy_killed, current_idx):

    tt, cc = spike_index[current_idx]
    energy_threshold = max(energy_killed[current_idx], energy[current_idx])
    ch_idx = np.where(neighbors[cc])[0]
    w = 5
    indices = temp[tt-w:tt+w+1, ch_idx].ravel()
    indices = indices[indices > -1]

    for j in indices:
        if energy[j] < energy_threshold:
            kill[j] = 1
            energy_killed[j] = energy_threshold

    return kill, energy_killed

#def kill_spikes_old(temp, neighbors, spike_index, energy, kill,
                #energy_killed, current_idx):

    #tt, cc = spike_index[current_idx]
    #energy_threshold = max(energy_killed[current_idx], energy[current_idx])
    #ch_idx = np.where(neighbors[cc])[0]
    #w = 5
    #indices = temp[tt-w:tt+w+1, ch_idx].ravel()
    #indices = indices[indices > -1]

    #for j in indices:
        #if energy[j] < energy_threshold:
            #kill[j] = 1
            #energy_killed[j] = energy_threshold

        #return kill, energy_killed

    #n_data = spike_index.shape[0]

    #temp = np.ones((T, C), 'int32')*-1
    #temp[spike_index[:, 0], spike_index[:, 1]] = np.arange(n_data)

    #kill = np.zeros(n_data, 'bool')
    #energy_killed = np.zeros(n_data, 'float32')
    #search_order = np.argsort(energy)[::-1]

    ## loop over all spikes
    #for j in range(n_data):
        #kill, energy_killed = kill_spikes(temp, neighbors, spike_index,
                                          #energy, kill,
                                          #energy_killed, search_order[j])

    #return kill



def fix_indexes_firstbatch(res, buffer_size, chunk_len, offset):
    
    """Fixes indexes from the first batch; 

    Parameters
    ----------
    res: tuple
        A tuple with the results from the nnet detector
    idx_local: slice
        A slice object indicating the indices for the data (excluding buffer)
    idx: slice
        A slice object indicating the absolute location of the data
    buffer_size: int
        Buffer size
    """
    score, clear, collision = res

    # get limits for the data (exlude indexes that have buffer data)
    data_start = buffer_size
    data_end = buffer_size + chunk_len

    # fix clear spikes
    clear_times = clear[:, 0]
    # get only observations outside the buffer
    idx_not_in_buffer = np.logical_and(clear_times >= data_start,
                                       clear_times <= data_end)
    clear_not_in_buffer = clear[idx_not_in_buffer]
    score_not_in_buffer = score[idx_not_in_buffer]

    # offset spikes depending on the absolute location
    clear_not_in_buffer[:, 0] = (clear_not_in_buffer[:, 0] + offset
                                 - buffer_size)

    # fix collided spikes
    col_times = collision[:, 0]
    # get only observations outside the buffer
    col_not_in_buffer = collision[np.logical_and(col_times >= data_start,
                                                 col_times <= data_end)]
    # offset spikes depending on the absolute location
    col_not_in_buffer[:, 0] = col_not_in_buffer[:, 0] + offset - buffer_size


    return score_not_in_buffer, clear_not_in_buffer, col_not_in_buffer



def fix_indexes_firstbatch_3(spike_index_list,  
                             offsets,
                             buffer_size, chunk_len):
            
    """ Fixes indexes from the first batch in the list of data with 
        multiple buffer offsets

    Parameters

    buffer_size: int
        Buffer size
    """

    # get limits for the data (exlude indexes that have buffer data)
    data_start = buffer_size
    data_end = buffer_size + chunk_len

    #fixed_scores = []
    #fixed_clear_spikes = []
    fixed_allspikes = []
    for (collision, offset) in zip(spike_index_list, offsets):

        ## ********** fix clear spikes **********
        ##clear_times = clear[:, 0]

        ## get only observations outside the buffer
        #idx_not_in_buffer = np.logical_and(clear_times >= data_start,
                                           #clear_times <= data_end)

        #clear_not_in_buffer = clear[idx_not_in_buffer]
        ##score_not_in_buffer = score[idx_not_in_buffer]

        ## offset spikes depending on the absolute location
        #clear_not_in_buffer[:, 0] = (clear_not_in_buffer[:, 0] + offset
                                     #- buffer_size)

        # ********** fix collided spikes *********
        col_times = collision[:, 0]
        
        # get only observations outside the buffer
        col_not_in_buffer = collision[np.logical_and(col_times >= data_start,
                                                     col_times <= data_end)]
        # offset spikes depending on the absolute location
        col_not_in_buffer[:, 0] = col_not_in_buffer[:, 0] + offset - buffer_size

        #fixed_scores.append(score_not_in_buffer)
        #fixed_clear_spikes.append(clear_not_in_buffer)
        fixed_allspikes.append(col_not_in_buffer)

    #return score_not_in_buffer, clear_not_in_buffer, col_not_in_buffer
    return np.vstack(fixed_allspikes)


def fix_indexes_firstbatch_2(spike_index_list, spike_index_clear_postkill, 
                             offsets,
                             buffer_size, chunk_len):
            
    """ Fixes indexes from the first batch in the list of data with 
        multiple buffer offsets

    Parameters

    buffer_size: int
        Buffer size
    """

    # get limits for the data (exlude indexes that have buffer data)
    data_start = buffer_size
    data_end = buffer_size + chunk_len

    #fixed_scores = []
    fixed_clear_spikes = []
    fixed_allspikes = []
    for (collision, clear, offset) in zip(spike_index_list, 
         spike_index_clear_postkill, offsets):

        # ********** fix clear spikes **********
        clear_times = clear[:, 0]

        # get only observations outside the buffer
        idx_not_in_buffer = np.logical_and(clear_times >= data_start,
                                           clear_times <= data_end)

        clear_not_in_buffer = clear[idx_not_in_buffer]
        #score_not_in_buffer = score[idx_not_in_buffer]

        # offset spikes depending on the absolute location
        clear_not_in_buffer[:, 0] = (clear_not_in_buffer[:, 0] + offset
                                     - buffer_size)

        # ********** fix collided spikes *********
        col_times = collision[:, 0]
        
        # get only observations outside the buffer
        col_not_in_buffer = collision[np.logical_and(col_times >= data_start,
                                                     col_times <= data_end)]
        # offset spikes depending on the absolute location
        col_not_in_buffer[:, 0] = col_not_in_buffer[:, 0] + offset - buffer_size

        #fixed_scores.append(score_not_in_buffer)
        fixed_clear_spikes.append(clear_not_in_buffer)
        fixed_allspikes.append(col_not_in_buffer)

    #return score_not_in_buffer, clear_not_in_buffer, col_not_in_buffer
    return (np.vstack(fixed_clear_spikes), 
            np.vstack(fixed_allspikes))


def remove_incomplete_waveforms(spike_index, spike_size, recordings_length):
    """

    Parameters
    ----------
    spikes: numpy.ndarray
        A 2D array of detected spikes as returned from detect.threshold

    Returns
    -------
    numpy.ndarray
        A new 2D array with some spikes removed. If the spike index is in a
        position (beginning or end of the recordings) where it is not possible
        to draw a complete waveform, it will be removed
    numpy.ndarray
        A boolean 1D array with True entries meaning that the index is within
        the valid range
    """
    max_index = recordings_length - 1 - spike_size
    min_index = spike_size
    include = np.logical_and(spike_index[:, 0] <= max_index,
                             spike_index[:, 0] >= min_index)
    return spike_index[include], include

