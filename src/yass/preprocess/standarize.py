"""
Recording standarization
"""
import multiprocess
import os.path

from yass.batch import BatchProcessor
from yass.util import check_for_files, ExpandPath, LoadFile

import numpy as np


@check_for_files(filenames=[ExpandPath('output_filename'),
                            LoadFile('output_filename', 'yaml')],
                 mode='extract', relative_to='output_path')
def standarize(path_to_data, dtype, n_channels, data_order,
               sampling_frequency, max_memory, output_path,
               output_dtype, output_filename='standarized.bin',
               if_file_exists='skip', processes='max'):
    """
    Standarize recordings in batches and write results to disk. Standard
    deviation is estimated using the first batch

    Parameters
    ----------
    path_to_data: str
        Path to recordings in binary format

    dtype: str
        Recordings dtype

    n_channels: int
        Number of channels in the recordings

    data_order: str
        Recordings order, one of ('channels', 'samples'). In a dataset with k
        observations per channel and j channels: 'channels' means first k
        contiguous observations come from channel 0, then channel 1, and so
        on. 'sample' means first j contiguous data are the first observations
        from all channels, then the second observations from all channels and
        so on

    sampling_frequency: int
        Recordings sampling frequency in Hz

    max_memory: str
        Max memory to use in each batch (e.g. 100MB, 1GB)

    output_path: str
        Where to store the standarized recordings

    output_dtype: str
        dtype  for standarized data

    output_filename: str, optional
        Filename for the output data, defaults to whitened.bin

    if_file_exists: str, optional
        One of 'overwrite', 'abort', 'skip'. If 'overwrite' it replaces the
        standarized data if it exists, if 'abort' if raise a ValueError
        exception if the file exists, if 'skip' if skips the operation if the
        file exists

    processes: str or int, optional
        Number of processes to use, if 'max', it uses all cores in the machine
        if a number, it uses that number of cores

    Returns
    -------
    standarized_path: str
        Path to standarized recordings

    standarized_params: dict
        A dictionary with the parameters for the standarized recordings
        (dtype, n_channels, data_order)
    """
    processes = multiprocess.cpu_count() if processes == 'max' else processes
    _output_path = os.path.join(output_path, output_filename)

    # init batch processor
    bp = BatchProcessor(path_to_data, dtype, n_channels, data_order,
                        max_memory)

    sd = standard_deviation(bp, sampling_frequency)

    def divide(rec):
        return np.divide(rec, sd)

    # apply transformation
    (standarized_path,
     standarized_params) = bp.multi_channel_apply(divide,
                                                  mode='disk',
                                                  output_path=_output_path,
                                                  cast_dtype=output_dtype,
                                                  processes=processes)

    return standarized_path, standarized_params


def standard_deviation(batch_processor, sampling_frequency,
                       preprocess_fn=None):
    """Estimate standard deviation using the first batch in a large file
    """
    # read a batch from all channels
    batches = batch_processor.multi_channel()
    first_batch = next(batches)

    if preprocess_fn:
        first_batch = preprocess_fn(first_batch)

    # estimate standard deviation using the first batch
    sd = _standard_deviation(first_batch, sampling_frequency)

    return sd


def _standard_deviation(rec, sampling_freq):
    """Determine standard deviation of noise in each channel

    Parameters
    ----------
    rec : matrix [length of recording, number of channels]

    sampling_freq : int
        the sampling rate (in Hz)

    Returns
    -------
    sd : vector [number of channels]
        standard deviation in each channel
    """

    # if the size of recording is long enough, only get middle 5 seconds of
    # data
    small_t = np.min((int(sampling_freq*5), rec.shape[0]))
    mid_T = int(np.ceil(rec.shape[0]/2))
    rec_temp = rec[int(mid_T-small_t/2):int(mid_T+small_t/2)]

    # find standard deviation using robust method
    sd = np.median(np.abs(rec_temp), 0)/0.6745
    return sd
