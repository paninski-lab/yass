"""
Detection pipeline
"""
import logging
import os.path
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
from yass.preprocess import whiten
from yass.geometry import n_steps_neigh_channels
from yass.util import file_loader, save_numpy_object


def run2(standarized_path, standarized_params,
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


    return run_neural_network2(standarized_path,
                              standarized_params,
                              whiten_filter,
                              output_directory,
                              if_file_exists,
                              save_results)

def run_neural_network2(standarized_path, standarized_params, 
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
    TMP_FOLDER = os.path.join(CONFIG.data.root_folder, output_directory)

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
        (x_tf, output_tf, NND,
         NNAE, NNT) = neuralnetwork.prepare_nn(CONFIG.channel_index,
                                           whiten_filter,
                                           detection_th,
                                           triage_th,
                                           detection_fname,
                                           ae_fname,
                                           triage_fname)

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
        
        logger.info("# of chunks: ", len(idx_list))
        
        logger.info (idx_list)
        
        # run tensorflow 
        processing_ctr = 0
        #chunk_ctr = 0
        
        # chunks to cycle over are 10 x as much as initial chosen data
        total_processing = len(idx_list)*n_sec_chunk

        # keep tensorflow open
        # save iteratively
        fname_detection = os.path.join(CONFIG.data.root_folder,'tmp/detect/')
        if os.path.exists(fname_detection)==False:
            os.mkdir(fname_detection)
        
        # set tensorflow verbosity level
        tf.logging.set_verbosity(tf.logging.ERROR)

        # open etsnrflow session
        with tf.Session() as sess:
            NND.saver.restore(sess, NND.path_to_detector_model)
            NNAE.saver_ae.restore(sess, NNAE.path_to_ae_model)
            NNT.saver.restore(sess, NNT.path_to_triage_model)

            # read chunks of data first:
            # read chunk of raw standardized data
            # Cat: TODO: don't save to lists, might want to use numpy arrays directl
            #print (os.path.join(fname_detection,"detect_"+
            #                      str(chunk_ctr).zfill(5)+'.npz'))

            # loop over 10sec or 60 sec chunks
            for chunk_ctr, idx in enumerate(idx_list): 
                if os.path.exists(os.path.join(fname_detection,"detect_"+
                                  str(chunk_ctr).zfill(5)+'.npz'))==True:
                                           continue

                # reset lists 
                spike_index_list = []
                idx_clean_list = []
                energy_list = []
                TC_list = []
                offset_list = []

                # load chunk of data
                standardized_recording = binary_reader(idx, buffer_size, path_to_standardized,
                                          n_channels, CONFIG.data.root_folder)
                
                # run detection on smaller chunks of data, e.g. 1 sec
                # Cat: TODO: add last bit at end in case short
                indexes = np.arange(0, standardized_recording.shape[0], sampling_rate)

                # run tensorflow over 1sec chunks in general
                for ctr, index in enumerate(indexes[:-1]): 
                    
                    # save absolute index of each subchunk
                    offset_list.append(idx[0]+indexes[ctr])
                    
                    data_temp = standardized_recording[indexes[ctr]:indexes[ctr+1]]
    
                    # store size of recordings in case at end of dataset.
                    TC_list.append(data_temp.shape)
                    
                    # run detect nn
                    score, spike_index, idx_clean = sess.run(
                        output_tf, feed_dict={x_tf: data_temp})
                    
                    # idx_clean is the indexes of clear spikes in all_spikes
                    spike_index_list.append(spike_index)
                    idx_clean_list.append(idx_clean)

                    # run AE nn; required for remove_axon function
                    # Cat: TODO: Do we really need this: can we get energy list faster?
                    rot = NNAE.load_rotation()
                    energy_ = np.ptp(np.matmul(score[:, :, 0], rot.T), axis=1)
                    energy_list.append(energy_)
                   
                    logger.info('processed chunk: %s/%s,  # spikes: %s', 
                      str(processing_ctr), str(total_processing), spike_index.shape)
            
                    processing_ctr+=1
            
                # save chunk of data in case crashes occur
                #np.save(os.path.join(fname_detection,"score_"+
                #                       str(idx[0]).zfill(5)),score_list)
                np.savez(os.path.join(fname_detection,"detect_"+
                                       str(chunk_ctr).zfill(5)),
                                       spike_index_list = spike_index_list,
                                       idx_clean_list = idx_clean_list,
                                       energy_list = energy_list, 
                                       TC_list = TC_list,
                                       offset_list = offset_list)
                
                # increase index for chunk
                #chunk_ctr+=1

        # load all saved data;
        #score_list = []
        spike_index_list = []
        idx_clean_list = []
        energy_list = []
        TC_list = []
        offset_list = []
        for ctr, idx in enumerate(idx_list): 
            data = np.load(fname_detection+'detect_'+str(ctr).zfill(5)+'.npz')
            spike_index_list.extend(data['spike_index_list'])
            idx_clean_list.extend(data['idx_clean_list'])
            energy_list.extend(data['energy_list'])
            TC_list.extend(data['TC_list'])
            offset_list.extend(data['offset_list'])

        # remove axons - compute axons to be killed
        logger.info(' removing axons in parallel')
        multi_procesing = 1
        if CONFIG.resources.multi_processing:
            killed = parmap.map(remove_axons_parallel,
                                list(zip(spike_index_list, energy_list, TC_list,
                                np.arange(len(energy_list)))),
                                neighbors,
                                processes=CONFIG.resources.n_processors,
                                pm_pbar=True)
        else:
            killed=[]
            for k in range(len(energy_list)):
                killed.append(remove_axons_parallel((spike_index_list[k], 
                                                     energy_list[k], 
                                                     TC_list[k],k),
                                                     neighbors))

        # Cat: TO DO - don't think there's a problem here
        ## make list of parmap returning order in case gets shuffled
        #killed_indexes = np.zeros(len(killed),'int32')
        #for k in range(len(killed)):
            #killed_indexes[k]=killed[k][1]

        # Cat: TODO Note that we're killing spike_index_all as well.

        # remove axons from clear spikes - keep only non-killed+clean events
        spike_index_all_postkill = []
        #score_clear_postkill = []
        spike_index_clear_postkill = []
        for k in range(len(idx_clean_list)):
            idx_keep = np.logical_and(~killed[k][0], idx_clean_list[k])
            #score_clear_postkill.append(score_list[k][idx_keep])
            spike_index_clear_postkill.append(spike_index_list[k][idx_keep])
            spike_index_all_postkill.append(spike_index_list[k][~killed[k][0]])

        # modified fix index file
        logger.info(' fixing indexes from batching')
        spike_index_clear, spike_index_all = fix_indexes_firstbatch_2(
                spike_index_all_postkill, spike_index_clear_postkill, 
                offset_list,
                buffer_size, sampling_rate)
                
        # get clear spikes
        clear = spike_index_clear

        #logger.info('Removing clear indexes outside the allowed range to '
        #            'draw a complete waveform...')
        _n_observations = fp_len
        clear, idx = detect.remove_incomplete_waveforms(
            clear, CONFIG.spike_size + CONFIG.templates.max_shift,
            _n_observations)

        # get scores for clear spikes
        #scores = score[idx]

        # get and clean all spikes
        spikes_all = spike_index_all 
        
        #logger.info('Removing all indexes outside the allowed range to '
        #            'draw a complete waveform...')
        spikes_all, _ = detect.remove_incomplete_waveforms(
            spikes_all, CONFIG.spike_size + CONFIG.templates.max_shift,
            _n_observations)

        #np.save(os.path.join(TMP_FOLDER,'scores_clear.npy'),scores)
        np.save(os.path.join(TMP_FOLDER,'spike_index_clear.npy'),clear)
        np.save(os.path.join(TMP_FOLDER,'spike_index_all.npy'),spikes_all)
    
    else:
        
        #scores = np.load(os.path.join(TMP_FOLDER,'scores_clear.npy'))
        clear = np.load(os.path.join(TMP_FOLDER,'spike_index_clear.npy'))
        spikes_all = np.load(os.path.join(TMP_FOLDER,'spike_index_all.npy'))

    #quit()

    return clear, spikes_all

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

