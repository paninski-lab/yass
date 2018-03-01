from __future__ import division
import os
import yaml
import numpy as np
from functools import partial, reduce
from collections import Iterable


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

    loader: str ('memmap', 'array' or 'python'), optional
        How to load the data. memmap loads the data using a wrapper around
        np.memmap (see :class:`~yass.batch.MemoryMap` for details), 'array'
        using numpy.fromfile and 'python' loads it using a wrapper
        around Python file API. Defaults to 'python'. Beware that the Python
        loader has limited indexing capabilities, see
        :class:`~yass.batch.BinaryReader` for details


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
                 data_format=None, loader='memmap'):

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

        self._data_format = data_format
        self._n_channels = n_channels
        self._dtype = dtype if not isinstance(dtype, str) else np.dtype(dtype)

        filesize = os.path.getsize(path_to_recordings)

        if not (filesize / self._dtype.itemsize).is_integer():
            raise ValueError('Wrong filesize and/or dtype, filesize {:, }'
                             'bytes is not divisible by the item size {}'
                             ' bytes'.format(filesize, self._dtype.itemsize))

        if int(filesize / self._dtype.itemsize) % n_channels:
            raise ValueError('Wrong n_channels, length of the data does not '
                             'match number of n_channels (observations % '
                             'n_channels != 0, verify that the number of '
                             'n_channels and/or the dtype are correct')

        self._n_observations = int(filesize / self._dtype.itemsize /
                                   n_channels)

        if loader not in ['memmap', 'array', 'python']:
            raise ValueError("loader must be one of 'memmap', 'array' or "
                             "'python'")

        order = dict(wide='F', long='C')

        shape = self._n_observations, n_channels

        def fromfile(path, dtype, data_format, shape):
            if data_format == 'long':
                return np.fromfile(path, dtype=dtype).reshape(shape)
            else:
                return np.fromfile(path, dtype=dtype).reshape(shape[::-1]).T

        if loader in ['memmap', 'array']:
            fn = (partial(MemoryMap, mode='r', shape=shape,
                          order=order[data_format])
                  if loader == 'memmap' else partial(fromfile,
                                                     data_format=data_format,
                                                     shape=shape))
            self._data = fn(path_to_recordings, dtype=self._dtype)

            if loader == 'array':
                self._data = self._data.reshape(shape)
        else:
            self._data = BinaryReader(path_to_recordings, dtype, shape,
                                      order=order[data_format])

    def __getitem__(self, key):

        # this happens when doung something like
        # x[[1,2,3]] or x[np.array([1,2,3])]
        if not isinstance(key, tuple):
            key = (key, slice(None))

        return self._data[key]

    def __repr__(self):
        return ('Reader for recordings with {:,} observations and {:,} '
                'channels in "{}" format'
                .format(self.observations, self.channels,
                        self._data_format))

    @property
    def shape(self):
        """Data shape in (observations, channels) format
        """
        return self._data.shape

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


class BinaryReader(object):
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

    def __init__(self, path_to_file, dtype, shape, order='F'):
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

    def _read_from_starts(self, f, starts):
        b = [self._read_n_bytes_from(f, n=1, start=s) for s in starts]
        return reduce(lambda x, y: x+y, b)

    def _read_row_major_order(self, rows, col_start, col_end):
        """Data where contiguous bytes are from the same row (C, row-major)
        """
        # compute offset to read from "col_start"
        start_byte = col_start * self.itemsize

        # number of consecutive observations to read
        n_cols_to_read = col_end - col_start

        # number of consecutive bytes to read
        to_read_bytes = n_cols_to_read * self.itemsize

        # compute bytes where reading starts in every row:
        # where row starts + offset due to row_start
        start_bytes = [row * self.row_size_byte + start_byte for row in rows]

        batch = [np.frombuffer(self._read_n_bytes_from(self.f, to_read_bytes,
                                                       start),
                               dtype=self.dtype)
                 for start in start_bytes]

        return np.array(batch)

    def _read_column_major_order(self, row_start, row_end, cols):
        """Data where contiguous bytes are from the same column
        (F, column-major)
        """
        # compute start byte position for every row
        start_byte = row_start * self.itemsize

        # how many consecutive bytes in each read
        rows_to_read = row_end - row_start
        to_read_bytes = self.itemsize * rows_to_read

        # compute seek poisitions (first "row_start "observation on
        # desired columns)
        start_bytes = [col * self.col_size_byte + start_byte for col in cols]

        batch = [np.frombuffer(self._read_n_bytes_from(self.f, to_read_bytes,
                                                       start),
                               dtype=self.dtype)
                 for start in start_bytes]
        batch = np.array(batch)

        return batch.T

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) > 2:
            raise ValueError('Must pass two slice objects i.e. obj[:,:]')

        int_key = any((isinstance(k, int) for k in key))

        def _int2slice(k):
            # when passing ints instead of slices array[0, 0]
            return k if not isinstance(k, int) else slice(k, k+1, None)

        key = [_int2slice(k) for k in key]

        for k in key:
            if isinstance(k, slice) and k.step:
                raise ValueError('Step size not supported')

        rows, cols = key

        # fill slices in case they are [:X] or [X:]
        if isinstance(rows, slice):
            rows = slice(rows.start or 0, rows.stop or self.n_row, None)

        if isinstance(cols, slice):
            cols = slice(cols.start or 0, cols.stop or self.n_col, None)

        if self.order == 'C':

            if isinstance(cols, Iterable):
                raise NotImplementedError('Column indexing with iterables '
                                          'is not implemented in C order')

            if isinstance(rows, slice):
                rows = range(rows.start, rows.stop)

            res = self._read_row_major_order(rows, cols.start, cols.stop)
        else:

            if isinstance(rows, Iterable):
                raise NotImplementedError('Row indexing with iterables '
                                          'is not implemented in F order')

            if isinstance(cols, slice):
                cols = range(cols.start, cols.stop)

            res = self._read_column_major_order(rows.start, rows.stop, cols)

        # convert to 1D array if either of keys was int
        return res if not int_key else res.reshape(-1)

    def __del__(self):
        self.f.close()

    @property
    def shape(self):
        return self.n_row, self.n_col

    def __len__(self):
        return self.n_row


# FIXME: this is a temporary solution, we need to investigate why memmap
# is blowing up memory
class MemoryMap:
    """
    Wrapper for numpy.memmap that creates a new memmap on each __getitem__
    call to save memory
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._init_mmap()

    def _init_mmap(self):
        self._mmap = np.memmap(*self.args, **self.kwargs)

    def __getitem__(self, index):
        res = self._mmap[index]
        self._init_mmap()
        return res

    def __getattr__(self, key):
        return getattr(self._mmap, key)
