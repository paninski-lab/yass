import numpy as np


def sd(rec, srate):
    """Determine standard deviation of noise in each channel

    Parameters
    ----------
    rec : matrix [length of recording, number of channels]
    srate : int
        the sampling rate (in Hz)

    Returns
    -------
    sd : vector [number of channels]
        standard deviation in each channel
    """

    # if the size of recording is long enough, only get middle 5 seconds of
    # data
    small_t = np.min((int(srate*5), rec.shape[0]))
    mid_T = int(np.ceil(rec.shape[0]/2))
    rec_temp = rec[int(mid_T-small_t/2):int(mid_T+small_t/2)]

    # find standard deviation using robust method
    sd = np.median(np.abs(rec_temp), 0)/0.6745

    return sd


def standarize(rec, _sd):
    """Standarize data
    """
    return np.divide(rec, _sd)
