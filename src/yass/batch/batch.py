"""
Batch processing
"""
from __future__ import division
import logging
from os import path

import numpy as np


class BatchProcessorFactory(object):
    """Object to create BatchProcessor objects

    Parameters
    ----------
    path_to_file: str
        Location of the binary file you want to process
    dtype: str
        dtype of your data, must be a valid numpy dtype
    n_channels:
        Number of channels in the recordings
    max_memory:
        Maximum memory for the process (bytes), the mimimum memory allowed
        is one observation per channel, e.g. if you have a dtype int16
        (2 bytes) and 50 channels, then the least you can load is 100
        bytes
    buffer_size: int
        Buffer size (on terms of observations)
    """
    def __init__(self, path_to_file, dtype, n_channels, max_memory,
                 buffer_size):
        self.path_to_file = path_to_file
        self.dtype = dtype
        self.n_channels = n_channels
        self.max_memory = max_memory
        self.buffer_size = buffer_size

    def make(self, **kwargs):
        """
        Create a new BatchProcessor instance with the selected configuration,
        if you do not pass any parameters, the BatchProcessor will be created
        with the parameters used to initialize the factory, otherwise the
        parameters will be replaced

        Parameters
        ----------
        **kwargs
            Any valid BatchProcessor initialization parameter
        """
        params = dict(path_to_file=self.path_to_file, dtype=self.dtype,
                      n_channels=self.n_channels, max_memory=self.max_memory,
                      buffer_size=self.buffer_size)
        params.update(kwargs)
        return BatchProcessor(**params)


class BatchProcessor(object):
    """Process data in batches

        Parameters
        ----------
        path_to_file: str
            Location of the binary file you want to process
        dtype: str
            dtype of your data, must be a valid numpy dtype
        n_channels:
            Number of channels in the recordings
        max_memory:
            Maximum memory for the process (bytes), the mimimum memory allowed
            is one observation per channel, e.g. if you have a dtype int16
            (2 bytes) and 50 channels, then the least you can load is 100
            bytes
        buffer_size: int
            Buffer size (on terms of observations)

        Examples
        --------

        .. literalinclude:: ../../examples/batch_processing.py
    """
    def __init__(self, path_to_file, dtype, n_channels, max_memory,
                 buffer_size):
        self.logger = logging.getLogger(__name__)

        self.path_to_file = path_to_file
        self.file = open(self.path_to_file, 'rb')
        self.dtype = dtype
        self.n_channels = n_channels
        self.dsize = np.dtype(dtype).itemsize
        self.buffer_size = buffer_size
        self.max_memory = max_memory

        # minimum quantity of data we can load (one observation per channel)
        observation_size_bytes = self.n_channels * self.dsize

        # size of file in bytes
        self.file_bytes = path.getsize(path_to_file)
        file_observations = self.file_bytes/observation_size_bytes

        if not float(file_observations).is_integer():
            raise ValueError('Invalid data size: n_channels ({}) * '
                             '[size of {}] ({}) must be a multiple of file '
                             'size ({})'.format(self.n_channels, self.dtype,
                                                self.dsize, self.file_bytes))
        else:
            self.file_observations = int(file_observations)

        # compute batch size (number of observations)
        if self.max_memory < observation_size_bytes:
            raise ValueError('Max memory should be at least {} bytes: '
                             '{} (dsize) * {} (n channels)'
                             .format(observation_size_bytes, self.dsize,
                                     self.n_channels))

        # number of observations to load in every batch
        batch_size = int(np.floor(self.max_memory/observation_size_bytes))
        # batch size in bytes
        batch_size_bytes = batch_size * self.dsize * self.n_channels

        if buffer_size > batch_size:
            raise ValueError('buffer size ({}) cannot be larger than '
                             'batch_size ({})'.format(buffer_size, batch_size))

        if batch_size_bytes >= self.file_bytes:
            self.n_batches = 1
            self.batch_size = self.file_observations
            self.residual = 0
            # self.n_portion = 1
        else:
            self.n_batches = int(np.ceil(float(self.file_observations)/batch_size))
            self.batch_size = batch_size
            self.residual = self.file_observations % batch_size
            # self.n_portion = np.ceil(self.preprocess.templates_partial_data * self.n_batches)

        self.i = 1

        self.logger.info('Number of batches: {}. Batch size: {} (observations '
                         ' per channel. Residual: {}. Total number of '
                         'observations {}'.format(self.n_batches,
                                                  self.batch_size,
                                                  self.residual,
                                                  self.file_observations))

    def __repr__(self):
        return ('BatchProcessor for file {} ({} bytes). Already processed {} '
                'out of {} batches. Max memory: {} bytes'
                .format(self.path_to_file, self.file_bytes,
                        self.i - 1,
                        self.n_batches, self.max_memory))

    def _add_zero_buffer(self, rec, size, option):
        """
        """
        buff = np.zeros((size, rec.shape[1]))

        if option == 'left':
            return np.append(buff, rec, axis=0)

        elif option == 'right':
            return np.append(rec, buff, axis=0)

        elif option == 'both':
            return np.concatenate((buff, rec, buff), axis=0)

    def _load_n(self, offset, n):
        """Load n observatons with an certain offset for every channel
        """
        self.file.seek(offset * self.dsize * self.n_channels)

        content = self.file.read(n * self.dsize * self.n_channels)

        rec = np.fromstring(content, dtype=self.dtype)
        rec = rec.reshape(n, self.n_channels)

        return rec

    def load_batch(self, i):
        """Load batch i

        Parameters
        ----------
        i: int
            Batch to load (1, 2, ..., n_batches)
        """
        self.logger.debug('Loading batch {}...'.format(i))

        if i > self.n_batches:
            raise ValueError('Cannot load batch {}, there are only {} batches '
                             'available'.format(i, self.n_batches))

        if self.n_batches == 1:
            rec = self._load_n(0, self.batch_size)
            rec = self._add_zero_buffer(rec, self.buffer_size, 'both')
            return rec
        else:
            # first batch
            if i == 1:
                self.logger.debug('Start position: 1. End position: {}'
                                  .format(self.batch_size + self.buffer_size))
                rec = self._load_n(0, self.batch_size + self.buffer_size)
                rec = self._add_zero_buffer(rec, self.buffer_size, 'left')
            # middle batch
            elif i < self.n_batches:
                start = (i - 1) * self.batch_size - self.buffer_size
                left = self.file_observations - i * self.batch_size

                # it may be the case that for some of the last batches, there
                # are not enough data left for the buffer size, so take at
                # most as you can
                size = (self.batch_size + self.buffer_size +
                        min(left, self.buffer_size))
                rec = self._load_n(start, size)

                # if buffer size is larger than the number of observations
                # left, fill with zer buffer to the right
                if self.buffer_size > left:
                    rec = self._add_zero_buffer(rec, self.buffer_size - left,
                                                'right')

                self.logger.debug('Start position: {}. End position: {}'
                                  .format(start + 1, start + size))
            # last batch
            else:
                if self.residual == 0:
                    rec = self._load_n((i - 1) * self.batch_size -
                                       self.buffer_size,
                                       self.buffer_size + self.batch_size)
                    rec = self._add_zero_buffer(rec, self.buffer_size, 'right')
                else:
                    rec = self._load_n((i - 1) * self.batch_size -
                                       self.buffer_size,
                                       self.buffer_size + self.residual)
                    size = self.buffer_size + (self.batch_size - self.residual)
                    rec = self._add_zero_buffer(rec, size, 'right')

        return rec

    def __iter__(self):
        return self

    def __next__(self):
        if self.i > self.n_batches:
            self.logger.debug('Closing file...')
            self.file.close()
            raise StopIteration('No more batches to process')

        batch = self.load_batch(self.i)
        self.i += 1

        return batch

    def next(self):
        # to make ir python 2 compatible...
        return self.__next__()

    def process_function(self, fn, path_to_file, *args, **kwargs):
        """Process a function in batches

        Parameters
        ----------
        fn: function
            Function to be run against every batch, the first argument in the
            function should be the argument to be sent in batches
        path_to_file: str
            Where to save the result
        *args, **args
            Other arguments t be passed to the function

        Examples
        --------

        .. literalinclude:: ../../examples/batch_processing_fn.py
        """
        f = open(path_to_file, 'wb')

        for batch in self:
            partial = fn(batch, *args, **kwargs)
            partial.tofile(f)

        f.close()

        return partial.dtype
