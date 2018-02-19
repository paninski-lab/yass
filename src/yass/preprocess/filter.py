"""
Filtering functions
"""
from yass.batch import BatchProcessor

from scipy.signal import butter, lfilter


def butterworth(path_to_data, dtype, n_channels, data_shape,
                low_frequency, high_factor, order, sampling_frequency,
                max_memory, output_path, output_dtype, if_file_exists='skip'):
    """Filter (butterworth) recordings in batches

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

    low_frequency: int
        Low pass frequency (Hz)

    high_factor: float
        High pass factor (proportion of sampling rate)

    order: int
        Order of Butterworth filter

    sampling_frequency: int
        Recordings sampling frequency in Hz

    max_memory: str
        Max memory to use in each batch (e.g. 100MB, 1GB)

    output_path: str
        Where to store the filtered recordings

    output_dtype: str
        dtype for filtered data

    if_file_exists: str, optional
        One of 'overwrite', 'abort', 'skip'. If 'overwrite' it replaces the
        file if it exists, if 'abort' if raise a ValueError exception if
        the file exists, if 'skip' if skips the operation if the file
        exists

    Returns
    -------
    standarized_path: str
        Location to filtered recordings

    standarized_params: dict
        A dictionary with the parameters for the filtered recordings
        (dtype, n_channels, data_format)
    """
    # init batch processor
    bp = BatchProcessor(path_to_data, dtype, n_channels, data_shape,
                        max_memory)

    (path,
     params) = bp.single_channel_apply(_butterworth, mode='disk',
                                       output_path=output_path,
                                       force_complete_channel_batch=True,
                                       if_file_exists=if_file_exists,
                                       cast_dtype=output_dtype,
                                       low_frequency=low_frequency,
                                       high_factor=high_factor,
                                       order=order,
                                       sampling_frequency=sampling_frequency)

    return path, params


def _butterworth(ts, low_frequency, high_factor, order, sampling_frequency):
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
    if ts.ndim > 1:
        raise NotImplementedError('This function can only be applied to a one'
                                  ' dimensional array, to apply it to '
                                  'multiple channels use the BatchProcessor')

    (T,) = ts.shape
    low = float(low_frequency)/sampling_frequency * 2
    high = float(high_factor) * 2
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, ts)
