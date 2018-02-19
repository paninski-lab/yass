"""
Recording standarization
"""
import os.path

from yass.batch import BatchProcessor
from yass.util import check_for_files, ExpandPath, LoadFile

import numpy as np


@check_for_files(parameters=['output_filename'],
                 if_skip=[ExpandPath('output_filename'),
                          LoadFile('output_filename', 'yaml')])
def standarize(path_to_data, dtype, n_channels, data_shape,
               sampling_frequency, max_memory, output_path,
               output_dtype, output_filename='standarized.bin',
               if_file_exists='skip'):
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

    data_shape: str
        Data shape, can be either 'long' (observations, channels) or
        'wide' (channels, observations)

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

    Returns
    -------
    standarized_path: str
        Path to standarized recordings

    standarized_params: dict
        A dictionary with the parameters for the standarized recordings
        (dtype, n_channels, data_format)
    """
    # init batch processor
    bp = BatchProcessor(path_to_data, dtype, n_channels, data_shape,
                        max_memory)

    # read a batch from all channels
    batches = bp.multi_channel()
    first_batch, _, _ = next(batches)

    # estimate standard deviation using the first batch
    sd = standard_deviation(first_batch, sampling_frequency)

    _output_path = os.path.join(output_path, output_filename)

    # apply transformation
    (standarized_path,
     standarized_params) = bp.multi_channel_apply(_standarize,
                                                  mode='disk',
                                                  output_path=_output_path,
                                                  cast_dtype=output_dtype,
                                                  sd=sd)

    return standarized_path, standarized_params


def _standarize(rec, sampling_freq=None, sd=None):
    """Standarize recordings

    Parameters
    ----------
    sampling_freq: int, optional
        Recording sample frequency, if None, sd is required

    sd: float, optional
        Standard deviation, if None, will be calculated


    Returns
    -------
    numpy.ndarray
        Standarized recordings

    """
    if sd is None:
        sd = standard_deviation(rec, sampling_freq)

    return np.divide(rec, sd)


def standard_deviation(rec, sampling_freq):
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
