import numpy as np


class RecordingsReader(object):
    """
    Neural recordings reader, supports wide and long data. Uses numpy.memmap
    under the hood to reduce memory usage

    Parameters
    ----------
    path_to_recordings: str
        Path to recordings file

    dtype: str
        Numpy dtype

    channels: int
        Number of channels

    data_format: str
        Data format, it can be either 'long' (observations, channels) or
        'wide' (channels, observations)

    Raises
    ------
    ValueError
        If dimensions do not match according to the file size, dtype and
        number of channels

    Examples
    --------

    .. literalinclude:: ../../examples/reader.py
    """

    def __init__(self, path_to_recordings, dtype, channels,
                 data_format):
        self.data = np.memmap(path_to_recordings, dtype=dtype)
        self.data_format = data_format
        self._channels = channels

        if len(self.data) % channels:
            raise ValueError('Wrong dimensions, length of the data does not '
                             'match number of channels (observations % '
                             'channels != 0, verify that the number of '
                             'channels and/or the dtype are correct')

        self._observations = int(len(self.data)/channels)

        dim = ((channels, self._observations) if data_format == 'wide' else
               (self._observations, channels))

        self.data = self.data.reshape(dim)

    def __getitem__(self, key):
        key = key if self.data_format == 'long' else key[::-1]
        subset = self.data[key]
        return subset if self.data_format == 'long' else subset.T

    @property
    def shape(self):
        return (self.data.shape if self.data_format == 'long' else
                self.data.shape[::-1])

    @property
    def observations(self):
        return self._observations

    @property
    def channels(self):
        return self._channels
