"""
Filtering functions
"""
from scipy.signal import butter, lfilter


def butterworth(ts, low_freq, high_factor, order, sampling_freq):
    """Butterworth filter for a one dimensional time series

    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_freq: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_freq: int
        Sampling frequency (Hz)

    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use the BatchProcessor

    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """
    if ts.ndim > 1:
        raise NotImplementedError('This function can only be applied to a one'
                                  ' dimensional array, to apply it to '
                                  'multiple channels use the BatchProcessor')

    (T,) = ts.shape
    low = float(low_freq)/sampling_freq * 2
    high = float(high_factor) * 2
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, ts)
