import logging
import os
import numpy as np
import os.path

from scipy.signal import butter, lfilter, filtfilt
from yass.preprocess.batch.standarize import _standard_deviation


def filter_standardize(data_in, low_frequency, high_factor, order,
                       sampling_frequency, buffer_size, filename_dat,
                       n_channels, output_directory):
    """Butterworth filter for a one dimensional time series

    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_frequency: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_frequency: int
        Sampling frequency (Hz)

    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use butterworth

    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """
    logger = logging.getLogger(__name__)

    # Load data from disk; buffer as required:
    idx_list, chunk_idx = data_in[0], data_in[1]

    # prPurple("Loading data files for chunk: "+str(chunk_idx))

    # New indexes
    idx_start = idx_list[0]
    idx_stop = idx_list[1]

    data_start = idx_start
    data_end = idx_stop

    # *****************************************************************
    # ********** LOAD RAW RECORDING ***********************************
    # *****************************************************************

    with open(filename_dat, "rb") as fin:
        if data_start == 0:
            # Seek position and read N bytes
            recordings_1D = np.fromfile(
                fin,
                dtype='int16',
                count=(data_end + buffer_size) * n_channels)
            recordings_1D = np.hstack((np.zeros(
                buffer_size * n_channels, dtype='int16'), recordings_1D))
        else:
            fin.seek((data_start - buffer_size) * 2 * n_channels, os.SEEK_SET)
            recordings_1D = np.fromfile(
                fin,
                dtype='int16',
                count=((data_end - data_start + buffer_size * 2) * n_channels))

        # If at end of recording
        if len(recordings_1D) != ((data_end - data_start +
                                   buffer_size * 2) * n_channels):
            recordings_1D = np.hstack((recordings_1D,
                                       np.zeros(buffer_size * n_channels,
                                                dtype='int16')))

    fin.close()

    # Convert to 2D array
    recordings = recordings_1D.reshape(-1, n_channels)

    ts = recordings

    # ******************************************************************
    # *********** FILTER DATA ******************************************
    # ******************************************************************

    if ts.ndim == 1:
        logger.info("SINGLE CHANNEL FILTER NOTE AVAILABLE.... !")
        quit()
        (T, ) = ts.shape
        low = float(low_frequency) / sampling_frequency * 2
        high = float(high_factor) * 2
        b, a = butter(order, [low, high], btype='band')

        output = lfilter(b, a, ts)

    else:
        T, C = ts.shape

        low = float(low_frequency) / sampling_frequency * 2
        high = float(high_factor) * 2
        b, a = butter(order, low, btype='high', analog=False)

        output = np.zeros((T, C), 'float32')
        for c in range(C):
            output[:, c] = filtfilt(b, a, ts[:, c])

    # Fix indexes function
    res = output[buffer_size:data_end - data_start + buffer_size]

    # Standardize data
    sd = _standard_deviation(res, sampling_frequency)
    standardized = np.divide(res, sd)

    np.save(
        os.path.join(output_directory,
                     "filtered_files/standardized_" + str(chunk_idx).zfill(6)),
        standardized)

    return standardized.shape


def merge_filtered_files(output_directory):

    logger = logging.getLogger(__name__)

    path = os.path.join(output_directory, 'filtered_files')
    filenames = os.listdir(path)
    filenames_sorted = sorted(filenames)

    f_out = os.path.join(output_directory, "standarized.bin")
    logger.info('...saving standardized file: %s', f_out)

    f = open(f_out, 'wb')
    for fname in filenames_sorted:
        res = np.load(os.path.join(path, fname))
        res.tofile(f)
        os.remove(os.path.join(path, fname))
