import numbers
import logging

import yaml

from .generator import IndexGenerator
from .reader import RecordingsReader
from .buffer import BufferGenerator


class BatchProcessor(object):
    """
    Batch processing for large numpy matrices

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

    max_memory: int or str
        Max memory to use in each batch, interpreted as bytes if int,
        if string, it can be any of {N}KB, {N}MB or {N}GB

    buffer_size: int, optional
        Buffer size, defaults to 0

    Raises
    ------
    ValueError
        If dimensions do not match according to the file size, dtype and
        number of channels
    """

    def __init__(self, path_to_recordings, dtype, n_channels,
                 data_format, max_memory, buffer_size=0):
        self.data_format = data_format
        self.buffer_size = buffer_size
        self.reader = RecordingsReader(path_to_recordings, dtype, n_channels,
                                       data_format)
        self.indexer = IndexGenerator(self.reader.observations,
                                      self.reader.channels,
                                      dtype,
                                      max_memory)
        self.buffer_generator = BufferGenerator(self.reader.observations,
                                                self.data_format,
                                                buffer_size=buffer_size)

        self.logger = logging.getLogger(__name__)

    def single_channel(self, force_complete_channel_batch=True, from_time=None,
                       to_time=None, channels='all'):
        """
        Generate indexes where each index has observations from a single
        channel

        Returns
        -------
        A generator that yields indexes

        Examples
        --------

        .. literalinclude:: ../../examples/batch/single_channel.py
        """
        indexes = self.indexer.single_channel(force_complete_channel_batch,
                                              from_time, to_time,
                                              channels)
        if force_complete_channel_batch:
            for idx in indexes:
                yield self.reader[idx]
        else:
            for idx in indexes:
                channel_idx = idx[1]
                yield self.reader[idx], channel_idx

    def multi_channel(self, from_time=None, to_time=None, channels='all'):
        """
        Generate indexes where each index has observations from more than
        one channel

        Returns
        -------
        generator:
            A slice that yields indexes

        Examples
        --------
        .. literalinclude:: ../../examples/batch/multi_channel.py
        """
        indexes = self.indexer.multi_channel(from_time, to_time, channels)

        for idx in indexes:
            if self.buffer_size:
                (idx_new,
                 (buff_start, buff_end)) = (self.buffer_generator
                                            .update_key_with_buffer(idx))
                subset = self.reader[idx_new]
                yield self.buffer_generator.add_buffer(subset, buff_start,
                                                       buff_end)
            else:
                yield self.reader[idx]

    def single_channel_apply(self, function, mode, output_path=None,
                             force_complete_channel_batch=True,
                             from_time=None, to_time=None, channels='all',
                             **kwargs):
        """
        Apply a transformation where each batch has observations from a
        single channel

        Parameters
        ----------
        function: callable
            Function to be applied, must accept a 1D numpy array as its first
            parameter
        mode: str
            'disk' or 'memory', if 'disk', a binary file is created at the
            beginning of the operation and each partial result is saved
            (ussing numpy.ndarray.tofile function), at the end of the
            operation two files are generated: the binary file and a yaml
            file with some file parameters (useful if you want to later use
            RecordingsReader to read the file). If 'memory', partial results
            are kept in memory and returned as a list
        output_path: str, optional
            Where to save the output, required if 'disk' mode
        force_complete_channel_batch: bool, optional
            If True, every index generated will correspond to all the
            observations in a single channel, hence
            n_batches = n_selected_channels, defaults to True. If True
            from_time and to_time must be None
        from_time: int, optional
            Starting time, defaults to None
        to_time: int, optional
            Ending time, defaults to None
        channels: int, tuple or str, optional
            A tuple with the channel indexes or 'all' to traverse all channels,
            defaults to 'all'
        **kwargs
            kwargs to pass to function

        Examples
        --------

        .. literalinclude:: ../../examples/batch/single_channel_apply.py

        Notes
        -----
        When applying functions in 'disk' mode will incur in memory overhead,
        which depends on the function implementation, this is an important
        thing to consider if the transformation changes the data's dtype (e.g.
        converts int16 to float64), which means that a chunk of 1MB in int16
        will have a size of 4MB in float64. Take that into account when
        setting max_memory.

        For performance reasons in 'disk' mode, output data is in 'wide' format
        """
        if mode not in ['disk', 'memory']:
            raise ValueError('Mode should be disk or memory, received: {}'
                             .format(mode))

        if mode == 'disk' and output_path is None:
            raise ValueError('output_path is required in "disk" mode')

        if mode == 'disk':
            fn = self._single_channel_apply_disk
            return fn(function, output_path,
                      force_complete_channel_batch, from_time,
                      to_time, channels, **kwargs)
        else:
            fn = self._single_channel_apply_memory
            return fn(function, force_complete_channel_batch, from_time,
                      to_time, channels, **kwargs)

    def multi_channel_apply(self, function, mode, output_path=None,
                            from_time=None, to_time=None, channels='all',
                            **kwargs):
        """
        Apply a function where each batch has observations from more than
        one channel

        Parameters
        ----------
        function: callable
            Function to be applied, must accept a 2D numpy array in 'long'
            format as its first parameter (number of observations, number of
            channels)
        mode: str
            'disk' or 'memory', if 'disk', a binary file is created at the
            beginning of the operation and each partial result is saved
            (ussing numpy.ndarray.tofile function), at the end of the
            operation two files are generated: the binary file and a yaml
            file with some file parameters (useful if you want to later use
            RecordingsReader to read the file). If 'memory', partial results
            are kept in memory and returned as a list
        output_path: str, optional
            Where to save the output, required if 'disk' mode
        force_complete_channel_batch: bool, optional
            If True, every index generated will correspond to all the
            observations in a single channel, hence
            n_batches = n_selected_channels, defaults to True. If True
            from_time and to_time must be None
        from_time: int, optional
            Starting time, defaults to None
        to_time: int, optional
            Ending time, defaults to None
        channels: int, tuple or str, optional
            A tuple with the channel indexes or 'all' to traverse all channels,
            defaults to 'all'
        **kwargs
            kwargs to pass to function

        Examples
        --------

        .. literalinclude:: ../../examples/batch/multi_channel_apply.py

        Notes
        -----
        Applying functions will incur in memory overhead, which depends
        on the function implementation, this is an important thing to consider
        if the transformation changes the data's dtype (e.g. converts int16 to
        float64), which means that a chunk of 1MB in int16 will have a size
        of 4MB in float64. Take that into account when setting max_memory

        For performance reasons, outputs data in 'long' format.
        """
        if mode not in ['disk', 'memory']:
            raise ValueError('Mode should be disk or memory, received: {}'
                             .format(mode))

        if mode == 'disk' and output_path is None:
            raise ValueError('output_path is required in "disk" mode')

        if mode == 'disk':
            fn = self._multi_channel_apply_disk
            return fn(function, output_path, from_time, to_time, channels,
                      **kwargs)
        else:
            fn = self._multi_channel_apply_memory
            return fn(function, from_time, to_time, channels, **kwargs)

    def _single_channel_apply_disk(self, function, output_path,
                                   force_complete_channel_batch, from_time,
                                   to_time, channels, **kwargs):
        f = open(output_path, 'wb')

        self.reader.output_shape = 'wide'
        indexes = self.indexer.single_channel(force_complete_channel_batch,
                                              from_time, to_time,
                                              channels)
        for i, idx in enumerate(indexes):
            self.logger.debug('Processing channel {}...'.format(i))
            self.logger.debug('Reading batch...')
            subset = self.reader[idx]
            self.logger.debug('Applying function...')
            res = function(subset, **kwargs)
            self.logger.debug('Writing to disk...')
            res.tofile(f)

        dtype = str(res.dtype)

        if channels == 'all':
            n_channels = self.reader.channels
        elif isinstance(channels, numbers.Integral):
            n_channels = 1
        else:
            n_channels = len(channels)

        f.close()

        # save yaml file with params
        path_to_yaml = output_path.replace('.bin', '.yaml')

        params = dict(dtype=dtype, n_channels=n_channels, data_format='wide')

        with open(path_to_yaml, 'w') as f:
            self.logger.debug('Saving params...')
            yaml.dump(params, f)

        return output_path, params

    def _multi_channel_apply_disk(self, function, output_path, from_time,
                                  to_time, channels, **kwargs):
        f = open(output_path, 'wb')

        self.reader.output_shape = 'long'
        indexes = self.indexer.multi_channel(from_time, to_time, channels)

        for idx in indexes:
            res = function(self.reader[idx], **kwargs)
            res.tofile(f)

        dtype = str(res.dtype)

        f.close()

        if channels == 'all':
            n_channels = self.reader.channels
        elif isinstance(channels, numbers.Integral):
            n_channels = 1
        else:
            n_channels = len(channels)

        # save yaml file with params
        path_to_yaml = output_path.replace('.bin', '.yaml')

        params = dict(dtype=dtype, n_channels=n_channels, data_format='long')

        with open(path_to_yaml, 'w') as f:
            yaml.dump(params, f)

        return output_path, params

    def _single_channel_apply_memory(self, function,
                                     force_complete_channel_batch, from_time,
                                     to_time, channels, **kwargs):

        indexes = self.indexer.single_channel(force_complete_channel_batch,
                                              from_time, to_time,
                                              channels)
        results = []

        for i, idx in enumerate(indexes):
            self.logger.debug('Processing channel {}...'.format(i))
            self.logger.debug('Reading batch...')
            subset = self.reader[idx]
            self.logger.debug('Applying function...')
            res = function(subset, **kwargs)
            self.logger.debug('Appending partial result...')
            results.append(res)

        return results

    def _multi_channel_apply_memory(self, function, from_time, to_time,
                                    channels, **kwargs):

        indexes = self.indexer.multi_channel(from_time, to_time, channels)
        results = []

        for idx in indexes:
            res = function(self.reader[idx], **kwargs)
            results.append(res)

        return results
