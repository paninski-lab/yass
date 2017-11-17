import logging

import numpy as np

from . import IndexGenerator, RecordingsReader


class BatchProcessor(object):
    """
    Batch processing for large numpy matrices

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

    max_memory: int or str
        Max memory to use in each batch, interpreted as bytes if int,
        if string, it can be any of {N}KB, {N}MB or {N}GB

    Raises
    ------
    ValueError
        If dimensions do not match according to the file size, dtype and
        number of channels

    Examples
    --------
    """
    def __init__(self, path_to_recordings, dtype, channels,
                 data_format, max_memory):
        self.data_format = data_format
        self.reader = RecordingsReader(path_to_recordings, dtype, channels,
                                       data_format)
        self.indexer = IndexGenerator(self.reader.observations,
                                      self.reader.channels,
                                      dtype,
                                      max_memory)

        self.logger = logging.getLogger(__name__)

    def single_channel(self, force_complete_channel_batch=True, from_time=None,
                       to_time=None, channels='all'):
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
        indexes = self.indexer.multi_channel(from_time, to_time, channels)

        for idx in indexes:
            # print('index', idx, self.reader[idx].shape)
            yield self.reader[idx]

    def single_channel_apply(self, function, output_path, output_dtype,
                             force_complete_channel_batch=True,
                             from_time=None, to_time=None, channels='all',
                             **kwargs):
        """
        Notes
        -----
        Applying functions will incur in memory overhead, which depends
        on the function implementation, this is an important thing to consider
        if the transformation changes the data's dtype (e.g. converts int16 to
        float64), which means that a chunk of 1MB in int16 will have a size
        of 4MB in float64. Take that into account when setting max_memory
        """
        out = np.memmap(output_path, output_dtype, 'w+',
                        shape=self.reader.data.shape)

        indexes = self.indexer.single_channel(force_complete_channel_batch,
                                              from_time, to_time,
                                              channels)
        for i, idx in enumerate(indexes):
            self.logger.info('Processing index {}, {}...'.format(i, idx))
            # TODO: decide what to do with the flipped indexes...
            out_idx = idx if self.data_format == 'long' else idx[::-1]
            read = self.reader[idx]#function()
            self.logger.info('Read...')
            out[out_idx] = read
            self.logger.info('Assigned...')

        out.flush()

        return output_dtype, output_path

    def multi_channel_apply(self, function, output_path, output_dtype,
                            from_time=None, to_time=None, channels='all',
                            **kwargs):
        """
        Notes
        -----
        Applying functions will incur in memory overhead, which depends
        on the function implementation, this is an important thing to consider
        if the transformation changes the data's dtype (e.g. converts int16 to
        float64), which means that a chunk of 1MB in int16 will have a size
        of 4MB in float64. Take that into account when setting max_memory
        """
        out = np.memmap(output_path, output_dtype, 'w+',
                        shape=self.reader.data.shape)

        indexes = self.indexer.multi_channel(from_time, to_time, channels)

        for idx in indexes:
            # TODO: decide what to do with the flipped indexes...
            out_idx = idx if self.data_format == 'long' else idx[::-1]
            out[out_idx] = function(self.reader[idx])

        out.flush()

        return output_dtype, output_path
