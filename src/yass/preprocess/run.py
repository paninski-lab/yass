"""
Preprocess pipeline
"""
import logging
import os
import numpy as np
import parmap
import yaml

from yass import read_config
from yass.preprocess.util import *
from yass.reader import READER
from yass.reordering import reorder

def run(output_directory):
    """Preprocess pipeline: filtering, standarization and whitening filter

    This step (optionally) performs filtering on the data, standarizes it
    and computes a whitening filter. Filtering and standardized data are
    processed in chunks and written to disk.

    Parameters
    ----------
    output_directory: str
        where results will be saved

    Returns
    -------
    standardized_path: str
        Path to standardized data binary file

    standardized_params: str
        Path to standardized data parameters

    channel_index: numpy.ndarray
        Channel indexes

    whiten_filter: numpy.ndarray
        Whiten matrix

    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/:

    * ``filtered.bin`` - Filtered recordings
    * ``filtered.yaml`` - Filtered recordings metadata
    * ``standardized.bin`` - Standarized recordings
    * ``standardized.yaml`` - Standarized recordings metadata
    * ``whitening.npy`` - Whitening filter

    Everything is run on CPU.

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/preprocess.py
    """

    # **********************************************
    # *********** Initialize ***********************
    # **********************************************
    
    logger = logging.getLogger(__name__)

    # load config
    CONFIG = read_config()

    # raw data info
    filename_raw = os.path.join(CONFIG.data.root_folder,
                                CONFIG.data.recordings)
    dtype_raw = CONFIG.recordings.dtype
    n_channels = CONFIG.recordings.n_channels

    if not CONFIG.preprocess.apply_filter:
        return filename_raw, dtype_raw

    # if apply filter, get recording reader
    n_sec_chunk = CONFIG.resources.n_sec_chunk
    reader = READER(filename_raw, dtype_raw, CONFIG, n_sec_chunk)
    logger.info("# of chunks: {}".format(reader.n_batches))

    # make output directory
    if not os.path.exists(output_directory):
        logger.info('Creating temporary folder: {}'.format(output_directory))
        os.makedirs(output_directory)
    else:
        logger.info('Temporary folder {} already exists, output will be '
                    'stored there'.format(output_directory))

    # make output parameters
    standardized_path = os.path.join(output_directory, "standardized.bin")
    standardized_params = dict(
        dtype=CONFIG.preprocess.dtype,
        n_channels=n_channels)
    logger.info('Output dtype for transformed data will be {}'
            .format(CONFIG.preprocess.dtype))
    reorder_fname = os.path.join(output_directory, "reorder.npy")
    # Check if data already saved to disk and skip:
    if os.path.exists(standardized_path):
        if os.path.exists(reorder_fname):
           return standardized_path, standardized_params['dtype'], reorder_fname
        reorder.run(save_fname = reorder_fname, 
                               standardized_fname = standardized_path, 
                               CONFIG = CONFIG, 
                               n_sec_chunk = 5, 
                               dtype = CONFIG.preprocess.dtype)
        return standardized_path, standardized_params['dtype'], reorder_fname



    # **********************************************
    # *********** run filter & stdarize  ***********
    # **********************************************

    # get necessary parameters
    low_frequency = CONFIG.preprocess.filter.low_pass_freq
    high_factor = CONFIG.preprocess.filter.high_factor
    order = CONFIG.preprocess.filter.order
    sampling_rate = CONFIG.recordings.sampling_rate

    # estimate std from a small chunk
    chunk_5sec = 5*CONFIG.recordings.sampling_rate
    if CONFIG.rec_len < chunk_5sec:
        chunk_5sec = CONFIG.rec_len
    small_batch = reader.read_data(
        data_start=CONFIG.rec_len//2 - chunk_5sec//2,
        data_end=CONFIG.rec_len//2 + chunk_5sec//2)

    fname_mean_sd = os.path.join(
        output_directory, 'mean_and_standard_dev_value.npz')
    if not os.path.exists(fname_mean_sd):
        get_std(small_batch, sampling_rate,
                fname_mean_sd, CONFIG.preprocess.apply_filter,
                low_frequency, high_factor, order)
    # turn it off
    small_batch = None

    # Make directory to hold filtered batch files:
    filtered_location = os.path.join(output_directory, "filtered_files")
    if not os.path.exists(filtered_location):
        os.makedirs(filtered_location)

    # read config params
    multi_processing = CONFIG.resources.multi_processing
    if CONFIG.resources.multi_processing:
        n_processors = CONFIG.resources.n_processors
        parmap.map(
            filter_standardize_batch,
            [i for i in range(reader.n_batches)],
            reader,
            fname_mean_sd,
            CONFIG.preprocess.apply_filter,
            CONFIG.preprocess.dtype,
            filtered_location,
            low_frequency,
            high_factor,
            order,
            sampling_rate,
            processes=n_processors,
            pm_pbar=True)
    else:
        for batch_id in range(reader.n_batches):
            filter_standardize_batch(
                batch_id, reader, fname_mean_sd,
                CONFIG.preprocess.apply_filter,
                CONFIG.preprocess.dtype,
                filtered_location,
                low_frequency,
                high_factor,
                order,
                sampling_rate,
                )

    # Merge the chunk filtered files and delete the individual chunks
    merge_filtered_files(filtered_location, output_directory)

    # save yaml file with params
    path_to_yaml = standardized_path.replace('.bin', '.yaml')
    with open(path_to_yaml, 'w') as f:
        logger.info('Saving params...')
        yaml.dump(standardized_params, f)
    reorder.run(save_fname = reorder_fname, 
                               standardized_fname = standardized_path, 
                               CONFIG = CONFIG, 
                               n_sec_chunk = 5, 
                               dtype = CONFIG.preprocess.dtype)

    return standardized_path, standardized_params['dtype'], reorder_fname
