import numbers
import logging

import yaml

from .generator import IndexGenerator
from .reader import RecordingsReader


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

    buffer: int, optional
        Buffer size, defaults to 0

    Raises
    ------
    ValueError
        If dimensions do not match according to the file size, dtype and
        number of channels
    """
    def __init__(self, path_to_recordings, dtype, n_channels,
                 data_format, max_memory, buffer=0):
        self.data_format = data_format
        self.reader = RecordingsReader(path_to_recordings, dtype, n_channels,
                                       data_format)
        self.indexer = IndexGenerator(self.reader.observations,
                                      self.reader.channels,
                                      dtype,
                                      max_memory)

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
        A generator that yields indexes

        Examples
        --------

        .. literalinclude:: ../../examples/batch/multi_channel.py
        """
        indexes = self.indexer.multi_channel(from_time, to_time, channels)

        for idx in indexes:
            yield self.reader[idx]

    def single_channel_apply(self, function, output_path,
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

        output_path: str
            Where to save the output

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
        Applying functions will incur in memory overhead, which depends
        on the function implementation, this is an important thing to consider
        if the transformation changes the data's dtype (e.g. converts int16 to
        float64), which means that a chunk of 1MB in int16 will have a size
        of 4MB in float64. Take that into account when setting max_memory.

        For performance reasons, outputs data in 'wide' format.
        """
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

    def multi_channel_apply(self, function, output_path,
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

        output_path: str
            Where to save the output

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
