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
        Max memory to use, interpreted as bytes if int, if string, it can be
        any of {N}KB, {N}MB or {N}GB

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
        self.reader = RecordingsReader(path_to_recordings, dtype, channels,
                                       data_format)
        self.indexer = IndexGenerator(self.reader.observations,
                                      self.reader.channels,
                                      dtype,
                                      max_memory)

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
            yield self.reader[idx]
