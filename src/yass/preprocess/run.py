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


def run(output_directory, if_file_exists='skip'):
    """Preprocess pipeline: filtering, standarization and whitening filter

    This step (optionally) performs filtering on the data, standarizes it
    and computes a whitening filter. Filtering and standardized data are
    processed in chunks and written to disk.

    Parameters
    ----------
    if_file_exists: str, optional
        One of 'overwrite', 'abort', 'skip'. Control de behavior for every
        generated file. If 'overwrite' it replaces the files if any exist,
        if 'abort' it raises a ValueError exception if any file exists,
        if 'skip' it skips the operation (and loads the files) if any of them
        exist

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

    # make output directory
    if not os.path.exists(output_directory):
        logger.info('Creating temporary folder: {}'.format(output_directory))
        os.makedirs(output_directory)
    else:
        logger.info('Temporary folder {} already exists, output will be '
                    'stored there'.format(output_directory))

    # make output parameters
    OUTPUT_DTYPE = CONFIG.preprocess.dtype
    standardized_params = dict(
        dtype=OUTPUT_DTYPE,
        n_channels=CONFIG.recordings.n_channels,
        data_order=CONFIG.recordings.order)
    logger.info('Output dtype for transformed data will be {}'
            .format(OUTPUT_DTYPE))

    # Check if data already saved to disk and skip:
    standardized_path = os.path.join(output_directory, "standardized.bin")
    if if_file_exists == 'skip':
        if os.path.exists(standardized_path):
            return str(standardized_path), standardized_params

        
    # **********************************************
    # *********** run filter & stdarize  ***********
    # **********************************************
    
    # get necessary parameters
    low_frequency = CONFIG.preprocess.filter.low_pass_freq
    high_factor = CONFIG.preprocess.filter.high_factor
    order = CONFIG.preprocess.filter.order
    sampling_rate = CONFIG.recordings.sampling_rate
    

    # get data reader
    filename_raw = os.path.join(CONFIG.data.root_folder,
                                CONFIG.data.recordings)
    dtype_raw = CONFIG.recordings.dtype
    n_sec_chunk = CONFIG.resources.n_sec_chunk
    reader = READER(filename_raw, dtype_raw, n_sec_chunk, CONFIG)

    logger.info("# of chunks: {}".format(reader.n_batches))


    #get standard deviation using a small chunk of data
    chunk_5sec = 5*CONFIG.recordings.sampling_rate
    small_batch = reader.read_data(
        data_start=CONFIG.rec_len//2 - chunk_5sec,
        data_end=CONFIG.rec_len//2 + chunk_5sec)

    fname_sd = os.path.join(
        output_directory, 'standard_dev_value.npy')
    get_std(small_batch, low_frequency, high_factor, order,
            sampling_rate, fname_sd)

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
            low_frequency,
            high_factor,
            order,
            sampling_rate,
            fname_sd,
            filtered_location,
            processes=n_processors,
            pm_pbar=True)
    else:
        for batch_id in range(reader.n_batches):
            filter_standardize_batch(
                batch_id, reader, low_frequency,
                high_factor, order, sampling_rate,
                fname_sd, filtered_location)

    # Merge the chunk filtered files and delete the individual chunks
    merge_filtered_files(filtered_location, output_directory)

    # save yaml file with params
    path_to_yaml = standardized_path.replace('.bin', '.yaml')
    with open(path_to_yaml, 'w') as f:
        logger.info('Saving params...')
        yaml.dump(standardized_params, f)

    return str(standardized_path), standardized_params
