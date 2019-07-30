import logging
import numpy as np
import os
from tqdm import tqdm
import parmap

from yass import read_config
from yass.reader import READER
from yass.cluster.cluster import Cluster
from yass.cluster.util import (make_CONFIG2, 
                               partition_input,
                               gather_clustering_result,
                               load_align_waveforms,
                               nn_denoise_wf, denoise_wf)
from yass.neuralnetwork import Denoise

def run(fname_spike_index,
        fname_recording,
        recording_dtype,
        output_directory,
        raw_data=True,
        full_run=False,
        chunk_sec=None,
        fname_residual=None,
        residual_dtype=None,
        fname_templates_up=None,
        fname_spike_train_up=None):

    """Spike clustering

    Parameters
    ----------

    spike_index: numpy.ndarray (n_clear_spikes, 2), str or Path
        2D array with indexes for spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum). Or path to an npy file

    out_dir: str, optional
        Location to store/look for the generate spike train relative to
        config output directory

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
    logger = logging.getLogger(__name__)

    CONFIG = read_config()
    # get CONFIG2 for clustering
    # Cat: TODO: Edu said the CONFIG file can be passed as a dictionary
    CONFIG2 = make_CONFIG2(CONFIG)

    # output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # if the output exists and want to skip, just finish
    fname_templates = os.path.join(output_directory, 'templates.npy')
    fname_spike_train = os.path.join(output_directory, 'spike_train.npy')
    if os.path.exists(fname_templates):
        return fname_templates, fname_spike_train

    # run cluster on chunk of data
    if chunk_sec is not None:
        max_time = CONFIG.recordings.sampling_rate*chunk_sec
        logger.info("clustering initial {} seconds of chunk".format(chunk_sec))

    else:
        max_time = CONFIG.recordings.sampling_rate*CONFIG.rec_len
        logger.info("clustering on full chunk")

    # partition spike_idnex_chunk using the second column
    partition_dir = os.path.join(output_directory, 'spt_partition')
    units, fnames_input = partition_input(partition_dir, 
                                          max_time,
                                          fname_spike_index,
                                          CONFIG,
                                          fname_templates_up,
                                          fname_spike_train_up)

    # data reader
    reader_raw = READER(fname_recording,
                        recording_dtype,
                        CONFIG)
    if fname_residual is not None:
        reader_resid = READER(fname_residual,
                              residual_dtype,
                              CONFIG)
    else:
        reader_resid = None
        
    # load and align waveforms
    logger.info("load and align waveforms on local channels")
    fnames_input = load_align_waveforms(
        os.path.join(output_directory, 'wfs'),
        fnames_input,
        reader_raw,
        reader_resid,
        raw_data,
        CONFIG2)

    if CONFIG.neuralnetwork.apply_nn:
        logger.info("NN denoise")
        # load NN denoiser
        denoiser = Denoise(CONFIG.neuralnetwork.denoise.n_filters,
                           CONFIG.neuralnetwork.denoise.filter_sizes,
                           CONFIG.spike_size_nn)
        denoiser.load(CONFIG.neuralnetwork.denoise.filename)
        # denoise it
        nn_denoise_wf(fnames_input, denoiser, CONFIG.torch_devices)

    else:
        logger.info("denoise")
        denoise_wf(fnames_input)

    # save location for intermediate results
    tmp_save_dir = os.path.join(output_directory, 'cluster_result')
    if not os.path.exists(tmp_save_dir):
        os.makedirs(tmp_save_dir)

    # Cat: TODO: this parallelization may not be optimally asynchronous
    # make arg list first
    args_in = []
    for ctr, unit in enumerate(units):

        # check to see if chunk + channel already completed
        filename_postclustering = os.path.join(
            tmp_save_dir, "cluster_result_{}.npz".format(unit))

        # skip 
        if os.path.exists(filename_postclustering):
            continue
        args_in.append([raw_data,
                        full_run,
                        CONFIG2,
                        reader_raw,
                        reader_resid,
                        filename_postclustering,
                        fnames_input[ctr]])

    logger.info("starting clustering")
    if CONFIG.resources.multi_processing:
        parmap.map(Cluster, args_in, 
                   processes=CONFIG.resources.n_processors,
                   pm_pbar=True)

    else:
        with tqdm(total=len(args_in)) as pbar:
            for arg_in in args_in:
                Cluster(arg_in)
                pbar.update()


    # first gather clustering result
    fname_templates, fname_spike_train = gather_clustering_result(
        tmp_save_dir, output_directory)

    for fname in fnames_input:
        os.remove(fname)

    return fname_templates, fname_spike_train
