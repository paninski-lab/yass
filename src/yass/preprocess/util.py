"""
Filtering functions
"""
import logging
import os
import numpy as np

from scipy.signal import butter, filtfilt


def _butterworth(ts, low_frequency, high_factor, order, sampling_frequency):
    """Butterworth filter

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

    low = float(low_frequency) / sampling_frequency * 2
    high = float(high_factor) * 2
    b, a = butter(order, low, btype='high', analog=False)

    if ts.ndim == 1:
        return filtfilt(b, a, ts)
    else:
        T, C = ts.shape
        output = np.zeros((T, C), 'float32')
        for c in range(C):
            output[:, c] = filtfilt(b, a, ts[:, c])

        return output


def _standard_deviation(rec, centered=False):
    """Determine standard deviation of noise in each channel

    Parameters
    ----------
    rec : matrix [length of recording, number of channels]

    centered : bool
        if not standardized, center it

    Returns
    -------
    sd : vector [number of channels]
        standard deviation in each channel
    """

    # find standard deviation using robust method
    if not centered:
        rec = rec - np.median(rec, axis=0, keepdims=True)
    return np.median(np.abs(rec), 0)/0.6745


def _standardize(rec, sd=None, centered=False):
    """Determine standard deviation of noise in each channel

    Parameters
    ----------
    rec : matrix [length of recording, number of channels]
        recording

    sd : vector [number of chnanels,]
        standard deviation

    centered : bool
        if not standardized, center it

    Returns
    -------
    matrix [length of recording, number of channels]
        standardized recording
    """

    # find standard deviation using robust method
    if not centered:
        rec = rec - np.median(rec, axis=0, keepdims=True)
    
    if sd is None:
        sd = _standard_deviation(rec, centered=True)

    # standardize all channels with SD> 0.1 (Voltage?) units
    # Cat: TODO: ensure that this is actually correct for all types of channels
    idx1 = np.where(sd>=0.1)[0]
    rec[:,idx1] = np.divide(rec[:,idx1],sd[idx1])
    
    # zero out bad channels
    idx2 = np.where(sd<0.1)[0]
    rec[:,idx2]=0.
    
    return rec
    #return np.divide(rec, sd)


def filter_standardize_batch(batch_id, reader, low_frequency, high_factor, order,
                             sampling_frequency, fname_sd, out_dtype,
                             output_directory):
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

    # read a batch
    ts = reader.read_data_batch(batch_id, add_buffer=True)

    # filter
    ts = _butterworth(ts, low_frequency, high_factor,
                          order, sampling_frequency)
    ts = ts[reader.buffer:-reader.buffer]

    # standardize
    sd = np.load(fname_sd)
    ts = _standardize(ts, sd)
    
    # save
    fname = os.path.join(
        output_directory,
        "standardized_{}.npy".format(
            str(batch_id).zfill(6)))
    np.save(fname, ts.astype(out_dtype))


def get_std(ts, low_frequency, high_factor, order,
            sampling_frequency, fname):
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

    # filter
    ts = _butterworth(ts, low_frequency, high_factor,
                      order, sampling_frequency)

    # standardize
    sd = _standard_deviation(ts)
    
    # save
    np.save(fname, sd)


def merge_filtered_files(filtered_location, output_directory):

    logger = logging.getLogger(__name__)

    filenames = os.listdir(filtered_location)
    filenames_sorted = sorted(filenames)

    f_out = os.path.join(output_directory, "standardized.bin")
    logger.info('...saving standardized file: %s', f_out)

    f = open(f_out, 'wb')
    for fname in filenames_sorted:
        res = np.load(os.path.join(filtered_location, fname))
        res.tofile(f)
        os.remove(os.path.join(filtered_location, fname))
