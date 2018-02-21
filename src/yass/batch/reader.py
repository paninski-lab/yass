import os
import yaml
import numpy as np
from functools import partial


class RecordingsReader(object):
    """
    Neural recordings reader, supports wide and long data. If a file with the
    same name but yaml extension exists in the directory it looks for dtype,
    channels and data_format, otherwise you need to pass the parameters in the
    constructor

    Parameters
    ----------
    path_to_recordings: str
        Path to recordings file

    dtype: str
        Numpy dtype

    n_channels: int
        Number of channels

    data_format: str
        Data format, it can be either 'long' (observations, channels) or
        'wide' (channels, observations)

    mmap: bool
        Whether to read the data using numpy.mmap, otherwise it reads
        the data using numpy.fromfile

    output_shape: str, optional
        Output shape, if 'wide', all subsets will be returned in 'wide' format
        (even if the data is in 'long' format), if 'long', all subsets are
        returned in 'long' format (even if the data is in 'wide') format.
        Defaults to 'long'

    Raises
    ------
    ValueError
        If dimensions do not match according to the file size, dtype and
        number of channels

    Notes
    -----
    This is just an utility class to index binary files in a consistent way,
    it does not matter the shape of the file ('wide' or 'long'), indexing is
    performed in [observations, channels] format. This class is mainly used
    by other internal YASS classes to maintain a consistent indexing order.

    Examples
    --------

    .. literalinclude:: ../../examples/batch/reader.py
    """

    def __init__(self, path_to_recordings, dtype=None, n_channels=None,
                 data_format=None, mmap=True, output_shape='long'):

        path_to_yaml = path_to_recordings.replace('.bin', '.yaml')

        if (not os.path.isfile(path_to_yaml) and (dtype is None or
           n_channels is None or data_format is None)):
            raise ValueError('At least one of: dtype, channels or data_format '
                             'are None, this is only allowed when a yaml '
                             'file is present in the same location as '
                             'the bin file, but no {} file exists'
                             .format(path_to_yaml))
        elif (os.path.isfile(path_to_yaml) and dtype is None and
              n_channels is None and data_format is None):
            with open(path_to_yaml) as f:
                params = yaml.load(f)

            dtype = params['dtype']
            n_channels = params['n_channels']
            data_format = params['data_format']

        loader = partial(np.memmap, mode='r') if mmap else np.fromfile
        self.output_shape = output_shape
        self._data = loader(path_to_recordings, dtype=dtype)
        self._data_format = data_format
        self._n_channels = n_channels
        self._dtype = dtype

        if len(self._data) % n_channels:
            raise ValueError('Wrong dimensions, length of the data does not '
                             'match number of n_channels (observations % '
                             'n_channels != 0, verify that the number of '
                             'n_channels and/or the dtype are correct')

        self._n_observations = int(len(self._data)/n_channels)

        dim = ((n_channels, self._n_observations) if data_format == 'wide' else
               (self._n_observations, n_channels))

        self._data = self._data.reshape(dim)

    def __getitem__(self, key):

        # this happens when doung something like
        # x[[1,2,3]] or x[np.array([1,2,3])]
        if not isinstance(key, tuple):
            key = (key, slice(None))

        key = key if self._data_format == 'long' else key[::-1]
        subset = self._data[key]
        return subset if self.data_format == self.output_shape else subset.T

    def __repr__(self):
        return ('Reader for recordings with {:,} observations and {:,} '
                'channels in "{}" format'
                .format(self.observations, self.channels,
                        self._data_format))

    @property
    def shape(self):
        """Data shape in (observations, channels) format
        """
        return (self._data.shape if self._data_format == 'long' else
                self._data.shape[::-1])

    @property
    def observations(self):
        """Number of observations
        """
        return self._n_observations

    @property
    def channels(self):
        """Number of channels
        """
        return self._n_channels

    @property
    def data_format(self):
        """Data format
        """
        return self._data_format

    @property
    def dtype(self):
        """Numpy's dtype
        """
        return self._dtype

    @property
    def data(self):
        """Underlying numpy data
        """
        return self._data
