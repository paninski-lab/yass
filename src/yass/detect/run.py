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
    logger.info('removing axons in parallel (TODO: repartition  \
                chunks to match # of cpus)')

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
    #n_sec_chunk = CONFIG.resources.n_sec_chunk*CONFIG.resources.n_processors
    batch_length = CONFIG.resources.n_sec_chunk*10
    n_sec_chunk = 0.5
    print ("   batch length to (sec): ", batch_length, 
           " (longer increase speed a bit)")
    print ("   length of each seg (sec): ", n_sec_chunk)
    buffer = NND.waveform_length
    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    reader = READER(standardized_path,
                    standardized_params['dtype'],
                    CONFIG,
                    batch_length,
                    buffer,
                    chunk_sec)

    # number of processed chunks
    processing_ctr = 0
    n_mini_per_big_batch = int(np.ceil(batch_length/n_sec_chunk))
    
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
                "detect_" + str(batch_id).zfill(5) + '.npz')
                
                # output_directory,
                # "detect_batch_" + str(batch_id).zfill(5) + "_of_total_"
                # + str(reader.n_batches) + '.npz')
                
            if os.path.exists(fname):
                processing_ctr += n_mini_per_big_batch
                continue

            # get a bach of size n_sec_chunk
            # but partioned into smaller minibatches of 
            # size n_sec_chunk_gpu
            batched_recordings, minibatch_loc_rel = reader.read_data_batch_batch(
                batch_id,
                #CONFIG.detect.n_sec_chunk,
                n_sec_chunk,
                add_buffer=True)
            
            # offset for big batch
            batch_offset = reader.idx_list[batch_id, 0] - reader.buffer
            # location of each minibatch (excluding buffer)
            minibatch_loc = minibatch_loc_rel + batch_offset
            spike_index_list = []
            energy_list = []
            for j in range(batched_recordings.shape[0]):
                # print ("  batchid: ", batch_id, ",  index: ", j, 
                       # batched_recordings[j].shape)
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

            #if processing_ctr%100==0:
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

