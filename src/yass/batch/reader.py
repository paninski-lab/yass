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

        self.output_shape = output_shape
        self._data_format = data_format
        self._n_channels = n_channels
        self._dtype = dtype

        loader = partial(np.memmap, mode='r') if mmap else np.fromfile
        self._data = loader(path_to_recordings, dtype=dtype)

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


# TODO: add performance considerations when reading long data
class BinaryReader(object):

    def __init__(self, path_to_file, dtype, shape, order='C'):
        """
        Reading batches from large array binary files on disk, similar to
        numpy.memmap. It is essentially just a wrapper around Python
        files API to read through large array binary file using the
        array[:,:] syntax.

        Parameters
        ----------
        order: str
            Array order 'C' for 'Row-major order' or 'F' for
            'Column-major order'


        Notes
        -----
        https://en.wikipedia.org/wiki/Row-_and_column-major_order
        """
        if order not in ('C', 'F'):
            raise ValueError('order must be either "C" or "F"')

        self.order = order
        self.dtype = dtype if not isinstance(dtype, str) else np.dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.n_row, self.n_col = shape
        self.f = open(path_to_file, 'rb')
        self.row_size_byte = self.itemsize * self.n_col
        self.col_size_byte = self.itemsize * self.n_row

    def _read_n_bytes_from(self, f, n, start):
        f.seek(int(start))
        return f.read(n)

    def _read_row_major_order(self, row_start, row_end, col_start, col_end):
        """Data where contiguous bytes are from the same row
        """
        # number of consecutive observations to read
        n_cols_to_read = col_end - col_start
        # number of consecutive bytes to read
        to_read_bytes = n_cols_to_read * self.itemsize

        # compute bytes where reading starts in every row:
        # where row starts + offset due to row_start
        start_bytes = [r_start * self.row_size_byte +
                       col_start * self.itemsize
                       for r_start in range(row_start, row_end)]

        batch = [np.frombuffer(self._read_n_bytes_from(self.f, to_read_bytes,
                                                       start),
                               dtype=self.dtype)
                 for start in start_bytes]

        return np.array(batch)

    def _read_column_major_order(self, row_start, row_end, col_start, col_end):
        """Data where contiguous bytes are from the same column
        """
        n_cols_to_read = col_end - col_start

        # compute the byte size of going from the nth row from column
        # i - 1 to the nth observation of column i
        jump_size_byte = self.col_size_byte

        # compute start poisition (first row on first desired col)
        # = observation 0 in first selected column + offset to move to
        # first desired row in first selected column
        start_byte = (self.col_size_byte * col_start +
                      row_start * self.itemsize)

        # how many consecutive bytes
        rows_to_read = row_end - row_start
        to_read_bytes = self.itemsize * rows_to_read

        # compute seek poisitions (first observation on desired columns)
        start_bytes = [start_byte + jump_size_byte * offset for offset in
                       range(n_cols_to_read)]

        batch = [np.frombuffer(self._read_n_bytes_from(self.f, to_read_bytes,
                                                       start),
                               dtype=self.dtype)
                 for start in start_bytes]
        batch = np.array(batch)

        return batch.T

    def __getitem__(self, key):

        if not isinstance(key, tuple) or len(key) > 2:
            raise ValueError('Must pass two slice objects i.e. obj[:,:]')

        if any(s.step for s in key):
            raise ValueError('Step size not supported')

        _row, _col = key

        row = slice(_row.start or 0, _row.stop or self.n_row, None)
        col = slice(_col.start or 0, _col.stop or self.n_col, None)

        if self.order == 'C':
            return self._read_row_major_order(row.start, row.stop,
                                              col.start, col.stop)
        else:
            return self._read_column_major_order(row.start, row.stop,
                                                 col.start, col.stop)

    def __del__(self):
        self.f.close()

    @property
    def shape(self):
        return self.n_row, self.n_col

    def __len__(self):
        return self.n_row
