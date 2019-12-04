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
import torch.multiprocessing as mp
import parmap

from yass import read_config
from yass.reader import READER
from yass.neuralnetwork import Detect, Denoise
from yass.util import file_loader
from yass.threshold.detect import voltage_threshold
from yass.detect.deduplication import deduplicate_gpu, deduplicate
from yass.detect.output import gather_result
from yass.geometry import make_channel_index


def run(standardized_path, standardized_dtype,
        output_directory, run_chunk_sec='full'):
           
    """Execute detect step

    Parameters
    ----------
    standardized_path: str or pathlib.Path
        Path to standardized data binary file

    standardized_dtype: string
        data type of standardized data

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
    if CONFIG.neuralnetwork.apply_nn:
        run_neural_network(
            standardized_path,
            standardized_dtype,
            output_temp_files,
            run_chunk_sec=run_chunk_sec)

    else:
        run_voltage_treshold(standardized_path,
                     standardized_dtype,
                     output_temp_files)

    ##### gather results #####
    gather_result(fname_spike_index,
                  output_temp_files)

    return fname_spike_index


def run_neural_network(standardized_path, standardized_dtype,
                       output_directory, run_chunk_sec='full'):
                           
    """Run neural network detection
    """
    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)

    # load NN detector
    detector = Detect(CONFIG.neuralnetwork.detect.n_filters,
                      CONFIG.spike_size_nn,
                      CONFIG.channel_index, CONFIG)
    detector.load(CONFIG.neuralnetwork.detect.filename)

    # load NN denoiser
    denoiser = Denoise(CONFIG.neuralnetwork.denoise.n_filters,
                       CONFIG.neuralnetwork.denoise.filter_sizes,
                       CONFIG.spike_size_nn, CONFIG)
    denoiser.load(CONFIG.neuralnetwork.denoise.filename)

    # get data reader
    batch_length = CONFIG.resources.n_sec_chunk*CONFIG.resources.n_processors
    n_sec_chunk = CONFIG.resources.n_sec_chunk_gpu_detect
    print ("   batch length to (sec): ", batch_length, 
           " (longer increase speed a bit)")
    print ("   length of each seg (sec): ", n_sec_chunk)
    buffer = CONFIG.spike_size_nn
    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    reader = READER(standardized_path,
                    standardized_dtype,
                    CONFIG,
                    batch_length,
                    buffer,
                    chunk_sec)

    # neighboring channels
    channel_index_dedup = make_channel_index(
        CONFIG.neigh_channels, CONFIG.geom, steps=2)

    # threshold for neuralnet detection
    detect_threshold = CONFIG.detect.threshold

    # loop over each chunk
    batch_ids = np.arange(reader.n_batches)
    batch_ids_split = np.split(batch_ids, len(CONFIG.torch_devices))
    if False:
        processes = []
        for ii, device in enumerate(CONFIG.torch_devices):
            p = mp.Process(target=run_nn_detction_batch,
                           args=(batch_ids_split[ii], output_directory, reader, n_sec_chunk,
                                 detector, denoiser, channel_index_dedup,
                                 detect_threshold, device))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        run_nn_detction_batch(batch_ids, output_directory, reader, n_sec_chunk,
                                 detector, denoiser, channel_index_dedup,
                                 detect_threshold, device=CONFIG.resources.gpu_id)
        

def run_nn_detction_batch(batch_ids, output_directory,
                          reader, n_sec_chunk,
                          detector, denoiser,
                          channel_index_dedup,
                          detect_threshold,
                          device):

    detector = detector.to(device)
    denoiser = denoiser.to(device)

    for batch_id in batch_ids:
        # skip if the file exists
        fname = os.path.join(
            output_directory,
            "detect_" + str(batch_id).zfill(5) + '.npz')

        if os.path.exists(fname):
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
        spike_index_dedup_list = []
        for j in range(batched_recordings.shape[0]):
            # detect spikes and get wfs
            spike_index, wfs = detector.get_spike_times(
                torch.FloatTensor(batched_recordings[j]).to(device),
                threshold=detect_threshold)

            # denoise and take ptp as energy
            wfs = denoiser(wfs)[0]
            energy = (torch.max(wfs, 1)[0] - torch.min(wfs, 1)[0])

            # deduplicate
            spike_index_dedup = deduplicate_gpu(
                spike_index, energy,
                batched_recordings[j].shape,
                channel_index_dedup)

            # convert to numpy
            spike_index = spike_index.cpu().data.numpy()
            spike_index_dedup = spike_index_dedup.cpu().data.numpy()

            # update the location relative to the whole recording
            spike_index[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
            spike_index_dedup[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
            spike_index_list.append(spike_index)
            spike_index_dedup_list.append(spike_index_dedup)

        #if processing_ctr%100==0:
        print('batch : {},  # spikes: {}'.format(batch_id, len(spike_index)))

        # save result
        np.savez(fname,
                 spike_index=spike_index_list,
                 spike_index_dedup=spike_index_dedup_list,
                 minibatch_loc=minibatch_loc)


def run_voltage_treshold(standardized_path, standardized_dtype,
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
    buffer = CONFIG.spike_size
    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    reader = READER(standardized_path,
                    standardized_dtype,
                    CONFIG,
                    batch_length,
                    buffer,
                    chunk_sec)

    # number of processed chunks
    n_mini_per_big_batch = int(np.ceil(batch_length/n_sec_chunk))    
    total_processing = int(reader.n_batches*n_mini_per_big_batch)

    # neighboring channels
    channel_index = make_channel_index(
        CONFIG.neigh_channels, CONFIG.geom, steps=2)
    
    if CONFIG.resources.multi_processing:
        parmap.starmap(run_voltage_threshold_parallel, 
                       list(zip(np.arange(reader.n_batches))),
                       reader,
                       n_sec_chunk,
                       CONFIG.detect.threshold,
                       channel_index,
                       output_directory,
                       processes=CONFIG.resources.n_processors,
                       pm_pbar=True)                
    else:
        for batch_id in range(reader.n_batches):
            run_voltage_threshold_parallel(
                batch_id,
                reader,
                n_sec_chunk,
                CONFIG.detect.threshold,
                channel_index,
                output_directory)


def run_voltage_threshold_parallel(batch_id, reader, n_sec_chunk,
                                   threshold, channel_index,
                                   output_directory):

    # skip if the file exists
    fname = os.path.join(
        output_directory,
        "detect_" + str(batch_id).zfill(5) + '.npz')

    if os.path.exists(fname):
        return

    # get a bach of size n_sec_chunk
    # but partioned into smaller minibatches of 
    # size n_sec_chunk_gpu
    batched_recordings, minibatch_loc_rel = reader.read_data_batch_batch(
        batch_id,
        n_sec_chunk,
        add_buffer=True)

    # offset for big batch
    batch_offset = reader.idx_list[batch_id, 0] - reader.buffer
    # location of each minibatch (excluding buffer)
    minibatch_loc = minibatch_loc_rel + batch_offset
    spike_index_list = []
    spike_index_dedup_list = []
    for j in range(batched_recordings.shape[0]):
        spike_index, energy = voltage_threshold(
            batched_recordings[j], 
            threshold)

        # move to gpu
        spike_index = torch.from_numpy(spike_index)
        energy = torch.from_numpy(energy)

        # deduplicate
        spike_index_dedup = deduplicate_gpu(
            spike_index, energy,
            batched_recordings[j].shape,
            channel_index)

        # convert to numpy
        spike_index = spike_index.cpu().data.numpy()
        spike_index_dedup = spike_index_dedup.cpu().data.numpy()
        
        # update the location relative to the whole recording
        spike_index[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
        spike_index_dedup[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
        spike_index_list.append(spike_index)
        spike_index_dedup_list.append(spike_index_dedup)

    # save result
    np.savez(fname,
             spike_index=spike_index_list,
             spike_index_dedup=spike_index_dedup_list,
             minibatch_loc=minibatch_loc)
