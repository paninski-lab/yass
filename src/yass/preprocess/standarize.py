"""
Recording standarization
"""
import numpy as np


def standarize(rec, sampling_freq=None, sd=None):
    """Standarize recordings
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
