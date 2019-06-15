"""
Detection pipeline
"""
import logging
import os
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

import numpy as np
import torch
import parmap

from yass import read_config
from yass.reader import READER
from yass.neuralnetwork import Detect, Denoise
from yass.util import file_loader
from yass.threshold.detect import threshold
from yass.threshold.dimensionality_reduction import pca
from yass.detect.deduplication import run_deduplication
from yass.detect.output import gather_result

def run(standardized_path, standardized_params,
        output_directory, run_chunk_sec='full'):
           
    """Execute detect step

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
    depending on how pytorch is configured.

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
        run_neural_network(
            standardized_path,
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


def run_neural_network_pytorch(standardized_path, standardized_params,
                               output_directory, run_chunk_sec='full'):
                           
    """Run neural network detection
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    # load NN detector
    detector = Detect(CONFIG.neuralnetwork.detect.n_filters,
                      CONFIG.spike_size_nn,
                      CONFIG.channel_index)
    detector.load(CONFIG.neuralnetwork.detect.filename)
    
    # threshold for neuralnet detection
    detect_threshold = CONFIG.detect.threshold

    # load NN denoiser
    denoiser = Denoise(CONFIG.neuralnetwork.denoise.n_filters,
                       CONFIG.neuralnetwork.denoise.filter_sizes,
                       CONFIG.spike_size_nn)
    denoiser.load(CONFIG.neuralnetwork.denoise.filename)

    # get data reader
    #n_sec_chunk = CONFIG.resources.n_sec_chunk*CONFIG.resources.n_processors
    batch_length = CONFIG.resources.n_sec_chunk*10
    n_sec_chunk = 0.5
    print ("   batch length to (sec): ", batch_length, 
           " (longer increase speed a bit)")
    print ("   length of each seg (sec): ", n_sec_chunk)
    buffer = CONFIG.spike_size_nn
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

    # loop over each chunk
    for batch_id in range(reader.n_batches):

        # skip if the file exists
        fname = os.path.join(
            output_directory,
            "detect_" + str(batch_id).zfill(5) + '.npz')

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

            spike_index, wfs = detector.get_spike_times(
                batched_recordings[j], threshold=detect_threshold)

            # convert to numpy
            spike_index = spike_index.cpu().data.numpy()

            # update the location relative to the whole recording
            spike_index[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
            spike_index_list.append(spike_index)

            # denoise and take ptp as energy
            #energy_ = denoiser(wfs)[0].data.numpy().ptp(1)
            wfs = denoiser(wfs)[0]
            energy_ = (torch.max(wfs, 1)[0] - torch.min(wfs, 1)[0]).cpu().data.numpy()
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


def run_amplitude_threshold(standardized_path, standardized_params,
                            output_directory, run_chunk_sec='full'):
                           
    """Run detection that thresholds on amplitude
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    # get data reader
    #n_sec_chunk = CONFIG.resources.n_sec_chunk*CONFIG.resources.n_processors
    batch_length = CONFIG.resources.n_sec_chunk*10
    n_sec_chunk = 0.5
    print ("   batch length to (sec): ", batch_length, 
           " (longer increase speed a bit)")
    print ("   length of each seg (sec): ", n_sec_chunk)
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

    threshold = CONFIG.detect.amplitude_threshold
    spike_size = CONFIG.spike_sze

    # loop over each chunk
    for batch_id in range(reader.n_batches):

        # skip if the file exists
        fname = os.path.join(
            output_directory,
            "detect_" + str(batch_id).zfill(5) + '.npz')

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

            spike_index, energy_ = amplitude_threshold(
                batched_recordings[j], 
                threshold,
                spike_size)

            # update the location relative to the whole recording
            spike_index[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
            spike_index_list.append(spike_index)
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
