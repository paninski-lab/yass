"""
Detection pipeline
"""
import logging
import os
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import parmap

from yass import read_config
from yass.reader import READER
from yass.neuralnetwork import NeuralNetDetector, AutoEncoder
from yass.util import file_loader, save_numpy_object, running_on_gpu
from yass.threshold.detect import threshold
from yass.threshold.dimensionality_reduction import pca
from yass.detect.deduplication import run_deduplication
from yass.detect.output import gather_result

def run(standardized_path, standardized_params,
        output_directory, run_chunk_sec='full'):
           
    """Execute detect step

    Cat: THIS CODE KEEP TENSORFLOW OPEN FOR DETECTION AND THEN COMPUTES 
         corrections post-detection

    Parameters
    ----------
    standardized_path: str or pathlib.Path
        Path to standardized data binary file

    standardized_params: dict, str or pathlib.Path
        Dictionary with standardized data parameters or path to a yaml file

    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/

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

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    # load files in case they are strings or Path objects
    standardized_params = file_loader(standardized_params)

    # make output directory if not exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    fname_spike_index = os.path.join(
        output_directory, 'spike_index.npy')
    if os.path.exists(fname_spike_index):
        return fname_spike_index

    ##### detection #####

    # save directory for temp files
    output_temp_files = os.path.join(
        output_directory, 'batch')
    if not os.path.exists(output_temp_files):
        os.mkdir(output_temp_files)

    # run detection
    if CONFIG.detect.method == 'threshold':
        run_threshold(standardized_path,
                      standardized_params,
                      output_temp_files)

    elif CONFIG.detect.method == 'nn':
        
        run_neural_network(standardized_path,
                           standardized_params,
                           output_temp_files,
                           run_chunk_sec=run_chunk_sec)

    ###### deduplication #####
    logger.info('removing axons in parallel')

    # save directory for deduplication
    dedup_output = os.path.join(
        output_directory, 'dedup')
    if not os.path.exists(dedup_output):
        os.mkdir(dedup_output)

    # run deduplication
    run_deduplication(output_temp_files,
                      dedup_output)

    ##### gather results #####
    gather_result(fname_spike_index,
                  output_temp_files,
                  dedup_output)

    return fname_spike_index
    
    
def run_neural_network(standardized_path, standardized_params,
                       output_directory, run_chunk_sec='full'):
                           
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

    # make tensorflow tensors and neural net classes
    detection_th = CONFIG.detect.neural_network_detector.threshold_spike
    detection_fname = CONFIG.detect.neural_network_detector.filename
    ae_fname = CONFIG.detect.neural_network_autoencoder.filename

    # open tensorflow for every chunk
    NND = NeuralNetDetector.load(detection_fname, detection_th,
                                 CONFIG.channel_index)
    NNAE = AutoEncoder.load(ae_fname, input_tensor=NND.waveform_tf)

    # get data reader
    n_sec_chunk = CONFIG.resources.n_sec_chunk*CONFIG.resources.n_processors

    buffer = NND.waveform_length
    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    reader = READER(standardized_path,
                    standardized_params['dtype'],
                    CONFIG,
                    n_sec_chunk,
                    buffer,
                    chunk_sec)

    # number of processed chunks
    processing_ctr = 0
    n_mini_per_big_batch = int(np.ceil(n_sec_chunk/CONFIG.detect.n_sec_chunk))
    total_processing = int(reader.n_batches*n_mini_per_big_batch)

    # set tensorflow verbosity level
    tf.logging.set_verbosity(tf.logging.ERROR)

    # extra params to disable memory hogging by tensorflow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config = tf.ConfigProto(device_count = {'GPU': 0}) 
    #sess = tf.Session(config=config)
    
    # open tensorflow session
    with tf.Session(config=config) as sess:

        NND.restore(sess)
        rot = NNAE.load_rotation(sess)

        # loop over each chunk
        for batch_id in range(reader.n_batches):

            # skip if the file exists
            fname = os.path.join(
                output_directory,
                "detect_"+str(batch_id).zfill(5)+'.npz')
            if os.path.exists(fname):
                processing_ctr += n_mini_per_big_batch
                continue

            # get a bach of size n_sec_chunk
            # but partioned into smaller minibatches of 
            # size n_sec_chunk_gpu
            batched_recordings, minibatch_loc_rel = reader.read_data_batch_batch(
                batch_id,
                CONFIG.detect.n_sec_chunk,
                add_buffer=True)
            
            # offset for big batch
            batch_offset = reader.idx_list[batch_id, 0] - reader.buffer
            # location of each minibatch (excluding buffer)
            minibatch_loc = minibatch_loc_rel + batch_offset
            spike_index_list = []
            energy_list = []
            for j in range(batched_recordings.shape[0]):
                spike_index, wfs = NND.predict_recording(
                    batched_recordings[j],
                    sess=sess,
                    output_names=('spike_index', 'waveform'))

                # update the location relative to the whole recording
                spike_index[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
                spike_index_list.append(spike_index)

                # run AE nn; required for remove_axon function
                # Cat: TODO: Do we really need this: can we get energy list faster?
                score = NNAE.predict(wfs, sess)
                energy_ = np.ptp(np.matmul(score[:, :, 0], rot.T), axis=1)
                energy_list.append(energy_)

                processing_ctr+=1

            logger.info('processed chunk: %s/%s,  # spikes: %s', 
                  str(processing_ctr), str(total_processing), len(spike_index))

            # save result
            np.savez(fname,
                     spike_index=spike_index_list,
                     energy=energy_list,
                     minibatch_loc=minibatch_loc)


# TODO: PETER, refactor threshold detector
def run_threshold(standardized_path, standardized_params,
        output_directory, temporal_features=3, std_factor=4):
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

    folder = output_directory
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
    clear = threshold(standardized_path,
                      standardized_params['dtype'],
                      standardized_params['n_channels'],
                      standardized_params['data_order'],
                      CONFIG.resources.max_memory,
                      CONFIG.neigh_channels,
                      CONFIG.spike_size,
                      CONFIG.spike_size + CONFIG.templates.max_shift,
                      std_factor,
                      TMP_FOLDER,
                      spike_index_clear_filename=filename_index_clear,
                      if_file_exists=if_file_exists)

    #######
    # PCA #
    #######

    # run PCA, save rotation matrix and pca scores under TMP_FOLDER
    # TODO: remove clear as input for PCA and create an independent function
    pca_scores, clear, _ = pca(standardized_path,
                               standardized_params['dtype'],
                               standardized_params['n_channels'],
                               standardized_params['data_order'],
                               clear,
                               CONFIG.spike_size,
                               temporal_features,
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
    scores = pca_scores

    # save spike_index_all (same as spike_index_clear for threshold
    # detector)
    path_to_spike_index_all = os.path.join(TMP_FOLDER,
                                           filename_spike_index_all)
    save_numpy_object(clear, path_to_spike_index_all, if_file_exists,
                      name='Spike index all')

    # FIXME: always saving scores since they are loaded by the clustering
    # step, we need to find a better way to do this, since the current
    # clustering code is going away soon this is a tmp solution
    # saves scores
    # saves whiten scores
    path_to_scores = os.path.join(TMP_FOLDER, filename_scores_clear)
    save_numpy_object(scores, path_to_scores, if_file_exists,
                      name='scores')


#def fix_indexes_firstbatch_3(spike_index_list,  
#                             offsets,
#                             buffer_size, chunk_len):
#            
#    """ Fixes indexes from the first batch in the list of data with 
#        multiple buffer offsets
#
#    Parameters
#
#    buffer_size: int
#        Buffer size
#    """

    # get limits for the data (exlude indexes that have buffer data)
#    data_start = buffer_size
#    data_end = buffer_size + chunk_len

    #fixed_scores = []
    #fixed_clear_spikes = []
#    fixed_allspikes = []
#    for (collision, offset) in zip(spike_index_list, offsets):

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
#        col_times = collision[:, 0]
        
        # get only observations outside the buffer
#        col_not_in_buffer = collision[np.logical_and(col_times >= data_start,
#                                                     col_times <= data_end)]
        # offset spikes depending on the absolute location
#        col_not_in_buffer[:, 0] = col_not_in_buffer[:, 0] + offset - buffer_size

        #fixed_scores.append(score_not_in_buffer)
        #fixed_clear_spikes.append(clear_not_in_buffer)
#        fixed_allspikes.append(col_not_in_buffer)

    #return score_not_in_buffer, clear_not_in_buffer, col_not_in_buffer
#    return np.vstack(fixed_allspikes)


#def remove_incomplete_waveforms(spike_index, spike_size, recordings_length):
#    """

#    Parameters
#    ----------
#    spikes: numpy.ndarray
#        A 2D array of detected spikes as returned from detect.threshold

#    Returns
#    -------
#    numpy.ndarray
#        A new 2D array with some spikes removed. If the spike index is in a
#        position (beginning or end of the recordings) where it is not possible
#        to draw a complete waveform, it will be removed
#    numpy.ndarray
#        A boolean 1D array with True entries meaning that the index is within
#        the valid range
#    """
#    max_index = recordings_length - 1 - spike_size
#    min_index = spike_size
#    include = np.logical_and(spike_index[:, 0] <= max_index,
#                             spike_index[:, 0] >= min_index)
#    return spike_index[include], include

#
# def run_threshold(standardized_path, standardized_params,
#                   whiten_filter, output_directory, if_file_exists,
#                   save_results):
#     """Run threshold detection and autoencoder dimensionality reduction
#
#     Returns
#     -------
#     scores
#       Scores for all spikes
#
#     spike_index_clear
#       Spike indexes for clear spikes
#
#     spike_index_all
#       Spike indexes for all spikes
#     """
#     logger = logging.getLogger(__name__)
#
#     CONFIG = read_config()
#     TMP_FOLDER = CONFIG.path_to_output_directory
#
#     # check if all scores, clear and collision spikes exist..
#     path_to_score = os.path.join(TMP_FOLDER, 'scores_clear.npy')
#     path_to_spike_index_clear = os.path.join(TMP_FOLDER,
#                                              'spike_index_clear.npy')
#     path_to_spike_index_all = os.path.join(TMP_FOLDER, 'spike_index_all.npy')
#     path_to_rotation = os.path.join(TMP_FOLDER, 'rotation.npy')
#
#     path_to_standardized = os.path.join(TMP_FOLDER, 'standardized.bin')
#
#     paths = [path_to_score, path_to_spike_index_clear, path_to_spike_index_all]
#     exists = [os.path.exists(p) for p in paths]
#
#     if (if_file_exists == 'overwrite' or not any(exists)):
#
#         n_channels = CONFIG.recordings.n_channels
#
#         # run nn preprocess batch-wsie
#         neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)
#
#         # compute len of recording
#         filename_dat = os.path.join(CONFIG.data.root_folder,
#                                     CONFIG.data.recordings)
#         fp = np.memmap(filename_dat, dtype='int16', mode='r')
#         fp_len = fp.shape[0] / n_channels
#
#         # compute batch indexes
#         buffer_size = 200       # Cat: to set this in CONFIG file
#         sampling_rate = CONFIG.recordings.sampling_rate
#         #n_sec_chunk = CONFIG.resources.n_sec_chunk
#
#         # Cat: TODO: Set a different size chunk for clustering vs. detection
#         n_sec_chunk = 60
#
#         # take chunks
#         indexes = np.arange(0, fp_len, sampling_rate * n_sec_chunk)
#
#         # add last bit of recording if it's shorter
#         if indexes[-1] != fp_len :
#             indexes = np.hstack((indexes, fp_len ))
#
#         idx_list = []
#         for k in range(len(indexes) - 1):
#             idx_list.append([
#                 indexes[k], indexes[k + 1], buffer_size,
#                 indexes[k + 1] - indexes[k] + buffer_size
#             ])
#
#         idx_list = np.int64(np.vstack(idx_list))
#
#         idx_list = idx_list
#
#         logger.info("# of chunks: %i", len(idx_list))
#
#         logger.info (idx_list)
#
#         # run tensorflow
#         processing_ctr = 0
#         #chunk_ctr = 0
#
#         # chunks to cycle over are 10 x as much as initial chosen data
#         total_processing = len(idx_list)*n_sec_chunk
#
#         # keep tensorflow open
#         # save iteratively
#         fname_detection = os.path.join(CONFIG.path_to_output_directory,
#                                        'detect')
#         if not os.path.exists(fname_detection):
#             os.mkdir(fname_detection)
#
#         # loop over 10sec or 60 sec chunks
#         for chunk_ctr, idx in enumerate(idx_list):
#             if os.path.exists(os.path.join(fname_detection,"detect_"+
#                               str(chunk_ctr).zfill(5)+'.npz')):
#                                        continue
#
#             # reset lists
#             spike_index_list = []
#             #idx_clean_list = []
#             energy_list = []
#             TC_list = []
#             offset_list = []
#
#             # load chunk of data
#             standardized_recording = binary_reader(idx, buffer_size, path_to_standardized,
#                                       n_channels, CONFIG.data.root_folder)
#
#             # run detection on smaller chunks of data, e.g. 1 sec
#             # Cat: TODO: add last bit at end in case short
#             indexes = np.arange(0, standardized_recording.shape[0], int(sampling_rate))
#
#             # run tensorflow over 1sec chunks in general
#             for ctr, index in enumerate(indexes[:-1]):
#
#                 # save absolute index of each subchunk
#                 offset_list.append(idx[0]+indexes[ctr])
#
#                 data_temp = standardized_recording[indexes[ctr]:indexes[ctr+1]]
#
#                 # store size of recordings in case at end of dataset.
#                 TC_list.append(data_temp.shape)
#
#                 # run detect nn
#                 threshold = -3
#                 spike_size = 30
#                 res = _threshold2(data_temp, spike_size, threshold)
#                 spike_index, wfs = res
#                 spike_index_list.append(spike_index)
#
#                 pca = PCA(n_components=3)
#                 score = pca.fit_transform(wfs)
#                 energy_ = pca.inverse_transform(score).ptp(1)
#                 energy_list.append(energy_)
#
#                 logger.info('processed chunk: %s/%s,  # spikes: %s',
#                   str(processing_ctr), str(total_processing), spike_index.shape)
#
#                 processing_ctr+=1
#
#             # save chunk of data in case crashes occur
#             np.savez(os.path.join(fname_detection,"detect_"+
#                                    str(chunk_ctr).zfill(5)),
#                                    spike_index_list = spike_index_list,
#                                    energy_list = energy_list,
#                                    TC_list = TC_list,
#                                    offset_list = offset_list)
#
#         # load all saved data;
#         spike_index_list = []
#         energy_list = []
#         TC_list = []
#         offset_list = []
#         for ctr, idx in enumerate(idx_list):
#             data = np.load(fname_detection+'/detect_'+str(ctr).zfill(5)+'.npz')
#             spike_index_list.extend(data['spike_index_list'])
#             energy_list.extend(data['energy_list'])
#             TC_list.extend(data['TC_list'])
#             offset_list.extend(data['offset_list'])
#
#         # save all detected spikes pre axon_kill
#         spike_index_all_pre_deduplication = fix_indexes_firstbatch_3(
#                 spike_index_list,
#                 offset_list,
#                 buffer_size,
#                 sampling_rate)
#         np.save(os.path.join(TMP_FOLDER,'spike_index_all_pre_deduplication.npy'),
#                              spike_index_all_pre_deduplication)
#
#
#         # remove axons - compute axons to be killed
#         logger.info(' removing axons in parallel')
#         multi_procesing = 1
#         if CONFIG.resources.multi_processing:
#             keep = parmap.map(deduplicate,
#                                 list(zip(spike_index_list,
#                                 energy_list,
#                                 TC_list,
#                                 np.arange(len(energy_list)))),
#                                 neighbors,
#                                 processes=CONFIG.resources.n_processors,
#                                 pm_pbar=True)
#         else:
#             keep=[]
#             for k in range(len(energy_list)):
#                 keep.append(deduplicate((spike_index_list[k],
#                                                      energy_list[k],
#                                                      TC_list[k],k),
#                                                      neighbors))
#
#
#         # Cat: TODO Note that we're killing spike_index_all as well.
#         # remove axons from clear spikes - keep only non-killed+clean events
#         spike_index_all_postkill = []
#         for k in range(len(spike_index_list)):
#             spike_index_all_postkill.append(spike_index_list[k][keep[k][0]])
#
#         logger.info(' fixing indexes from batching')
#         spike_index_all = fix_indexes_firstbatch_3(
#                 spike_index_all_postkill,
#                 offset_list,
#                 buffer_size,
#                 sampling_rate)
#
#         # get and clean all spikes
#         spikes_all = spike_index_all
#
#         #logger.info('Removing all indexes outside the allowed range to '
#         #            'draw a complete waveform...')
#         _n_observations = fp_len
#         spikes_all, _ = detect.remove_incomplete_waveforms(
#             spikes_all, CONFIG.spike_size + CONFIG.templates.max_shift,
#             _n_observations)
#
#         np.save(os.path.join(TMP_FOLDER,'spike_index_all.npy'),spikes_all)
#
#     else:
#         spikes_all = np.load(os.path.join(TMP_FOLDER,'spike_index_all.npy'))
#
#     return spikes_all
