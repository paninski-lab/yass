import logging
import os.path
import os
import numpy as np
import parmap
import yaml

from yass.geometry import make_channel_index
from yass.preprocess.experimental.util import (filter_standardize,
                                               merge_filtered_files)
from yass.util import save_numpy_object
from yass.preprocess.batch import whiten


def run(CONFIG, if_file_exists='skip',
        filter_order=3, filter_low_pass_freq=300,
        filter_high_factor=0.1, multi_processing=True,
        n_sec_chunk=5):
    """Preprocess pipeline: filtering, standarization and whitening filter
    """

    logger = logging.getLogger(__name__)

    TMP = os.path.join(CONFIG.path_to_output_directory)
    output_directory = os.path.join(TMP, 'preprocess')

    if not os.path.exists(output_directory):
        logger.info('Creating temporary folder: {}'.format(output_directory))
        os.makedirs(output_directory)
    else:
        logger.info('Temporary folder {} already exists, output will be '
                    'stored there'.format(output_directory))

    params = dict(
        dtype=CONFIG.recordings.dtype,
        n_channels=CONFIG.recordings.n_channels,
        data_order=CONFIG.recordings.order)

    # Generate params:
    standarized_path = os.path.join(output_directory, "standarized.bin")
    standarized_params = params
    standarized_params['dtype'] = 'float32'

    # Check if data already saved to disk and skip:
    if if_file_exists == 'skip':
        f_out = os.path.join(output_directory, "standarized.bin")
        if os.path.exists(f_out):

            channel_index = make_channel_index(CONFIG.neigh_channels,
                                               CONFIG.geom, 2)

            # Cat: this is redundant, should save to disk/not recompute
            whiten_filter = whiten.matrix(
                standarized_path,
                standarized_params['dtype'],
                standarized_params['n_channels'],
                standarized_params['data_order'],
                channel_index,
                CONFIG.spike_size,
                CONFIG.resources.max_memory,
                output_directory,
                output_filename='whitening.npy',
                if_file_exists=if_file_exists)

            path_to_channel_index = os.path.join(output_directory,
                                                 "channel_index.npy")

            return str(standarized_path), standarized_params, whiten_filter

    # read config params
    n_processors = CONFIG.resources.processes
    n_channels = CONFIG.recordings.n_channels
    sampling_rate = CONFIG.recordings.sampling_rate

    buffer_size = 200

    # compute len of recording
    filename_dat = os.path.join(CONFIG.data.root_folder,
                                CONFIG.data.recordings)
    fp = np.memmap(filename_dat, dtype='int16', mode='r')
    fp_len = fp.shape[0]

    # compute batch indexes
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

    logger.info("# of chunks: %i", len(idx_list))

    # Make directory to hold filtered batch files:
    filtered_location = os.path.join(output_directory, "filtered_files")
    logger.info(filtered_location)
    if not os.path.exists(filtered_location):
        os.makedirs(filtered_location)

    # filter and standardize in one step
    if multi_processing:
        parmap.map(
            filter_standardize,
            list(zip(idx_list, proc_indexes)),
            filter_low_pass_freq,
            filter_high_factor,
            filter_order,
            sampling_rate,
            buffer_size,
            filename_dat,
            n_channels,
            output_directory,
            processes=n_processors,
            pm_pbar=True)
    else:
        for k in range(len(idx_list)):
            filter_standardize([idx_list[k], k], filter_low_pass_freq,
                               filter_high_factor,
                               filter_order, sampling_rate, buffer_size,
                               filename_dat,
                               n_channels, output_directory)

    # Merge the chunk filtered files and delete the individual chunks
    merge_filtered_files(output_directory)

    # save yaml file with params
    path_to_yaml = standarized_path.replace('.bin', '.yaml')

    params = dict(
        dtype=standarized_params['dtype'],
        n_channels=standarized_params['n_channels'],
        data_order=standarized_params['data_order'])

    with open(path_to_yaml, 'w') as f:
        logger.info('Saving params...')
        yaml.dump(params, f)

    # TODO: this shoulnd't be done here, it would be better to compute
    # this when initializing the config object and then access it from there
    channel_index = make_channel_index(CONFIG.neigh_channels, CONFIG.geom, 2)

    # logger.info CONFIG.resources.max_memory
    # quit()
    # Cat: TODO: need to make this much smaller in size, don't need such
    # large batches
    # OLD CODE: compute whiten filter using batch processor
    # TODO: remove whiten_filter out of output argument

    whiten_filter = whiten.matrix(
        standarized_path,
        standarized_params['dtype'],
        standarized_params['n_channels'],
        standarized_params['data_order'],
        channel_index,
        CONFIG.spike_size,
        # CONFIG.resources.max_memory,
        '50MB',
        output_directory,
        output_filename='whitening.npy',
        if_file_exists=if_file_exists)

    path_to_channel_index = os.path.join(output_directory, 'channel_index.npy')
    save_numpy_object(
        channel_index,
        path_to_channel_index,
        if_file_exists=if_file_exists,
        name='Channel index')

    return str(standarized_path), standarized_params, whiten_filter
